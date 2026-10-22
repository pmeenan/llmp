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
Llm::GenerationSession::GenerationSession(Llm& model, Branch& branch,
                                          const GenerateOptions& options, Generation& out)
    : model_(model), branch_(branch), options_(options), out_(out) {}

Llm::GenerationSession::~GenerationSession() {
  base::Check(finished_, "an unfinished generation session was destroyed");
}

std::expected<std::unique_ptr<Llm::GenerationSession>, std::string> Llm::BeginGeneration(
    Branch& branch, const std::vector<float>& last, const GenerateOptions& options, Generation& out,
    bool resume) {
  CheckBranch(branch);
  // A resumed generation chooses nothing from `last`: after a restore of its
  // spilled state (no prefill) there are none.
  if (last.empty() && !resume) {
    return Error(std::format("{} has no prefill's logits to generate from", name_));
  }
  if (options.max_tokens == 0) {
    return Error("generation needs a positive token budget");
  }
  if (branch.generation_active_ || branch.prompt_session_ != nullptr) {
    return Error("the conversation already has an active session");
  }
  if (resume && (out.tokens.empty() || out.stopped || out.tokens.size() >= options.max_tokens ||
                 branch.history_.size() + 1 >= context_)) {
    return Error("a resumed generation needs an unfinished generation within the context");
  }
  branch.generation_active_ = true;
  auto session =
      std::unique_ptr<GenerationSession>(new GenerationSession(*this, branch, options, out));
  branch.generation_session_ = session.get();
  if (auto began = session->Begin(last, resume); !began) {
    session->Close();
    return std::unexpected(began.error());
  }
  return session;
}

bool Llm::GenerationReadyFor(const Branch& branch) const {
  CheckBranch(branch);
  return branch.generation_active_ && branch.generation_session_ &&
         !branch.generation_session_->done() && branch.generation_session_->initialized_ &&
         !branch.generation_session_->prepared_ && !branch.generation_session_->publishing_;
}

std::optional<std::uint32_t> Llm::RemainingGenerationRowsFor(const Branch& branch) const {
  CheckBranch(branch);
  if (!branch.generation_session_) return std::nullopt;
  if (branch.generation_session_->done()) return 0;
  if (!branch.generation_session_->initialized_ || branch.generation_session_->prepared_ ||
      branch.generation_session_->publishing_)
    return std::nullopt;
  const auto& session = *branch.generation_session_;
  const auto rows = session.options_.max_tokens > session.out_.tokens.size()
                        ? session.options_.max_tokens - session.out_.tokens.size()
                        : 0;
  if (rows > UINT32_MAX) return std::nullopt;
  return static_cast<std::uint32_t>(rows);
}

bool Llm::GenerationSession::IsStop(std::int32_t token) const {
  return !(options_.tool_sampling && options_.tool_sampling->ReasoningBoundary(token)) &&
         options_.stop &&
         (std::ranges::find(model_.stops_, token) != model_.stops_.end() ||
          std::ranges::find(options_.extra_stops, token) != options_.extra_stops.end());
}

bool Llm::GenerationSession::Report() {
  // The new tokens since the last call, to on_tokens: false ends the
  // generation.
  if (!options_.on_tokens) {
    return true;
  }
  std::size_t visible = std::min<std::size_t>(out_.tokens.size(), options_.max_tokens);
  if (out_.stopped && visible == out_.tokens.size() && visible > 0 &&
      !(options_.tool_sampling && options_.tool_sampling->Preserves(out_.tokens.back()))) {
    --visible;  // the stop token
  }
  const std::span<const std::int32_t> fresh =
      std::span<const std::int32_t>(out_.tokens).subspan(reported_, visible - reported_);
  reported_ = visible;
  return options_.on_tokens(fresh);
}

Status Llm::GenerationSession::Begin(const std::vector<float>& last, bool resume) {
  if (!model_.ReserveTokens(branch_, all_, all_charge_, branch_.history_.size() + 1)) {
    return Error("native generation history exceeds the execution budget");
  }
  branch_.sampling_.reset();
  branch_.tool_sampling_ = options_.tool_sampling.get();
  if (options_.sampling && options_.sampling->temperature > 0) {
    branch_.sampling_ = options_.sampling;
    branch_.seed_ = options_.seed;
  }
  if (resume) {
    // The anchor was chosen and reported before the preemption; the state
    // now holds exactly the tokens before it again. Sampling stays keyed by
    // absolute position, so the continuation draws as it would have.
    out_.cancelled = false;
    all_ = branch_.history_;
    all_.push_back(out_.tokens.back());
    position_ = static_cast<std::uint32_t>(branch_.history_.size());
    reported_ = std::min<std::size_t>(out_.tokens.size(), options_.max_tokens);
    start_ = Clock::now();
    initialized_ = true;
    return {};
  }
  if (options_.tool_sampling)
    options_.tool_sampling->Budget(options_.max_tokens, model_.context_ - branch_.history_.size());
  auto first = model_.Choose(branch_, last, branch_.history_.size());
  if (!first) {
    return std::unexpected(first.error());
  }
  if (options_.tool_sampling) {
    auto plan = options_.tool_sampling->Plan(*first);
    if (!plan) return std::unexpected(plan.error());
    *first = plan->tokens.front();
    options_.tool_sampling->Certain({plan->tokens.begin() + 1, plan->tokens.end()});
    if (auto accepted = options_.tool_sampling->Accept(*first); !accepted)
      return std::unexpected(accepted.error());
  }
  out_.tokens = {*first};
  if (options_.keep_logits) {
    out_.logits = {last};
  }
  out_.stopped = IsStop(out_.tokens.back());
  out_.cancelled = options_.on_logits && !options_.on_logits(*first, last);
  out_.cancelled = !Report() || out_.cancelled;
  // Every token so far, the anchor (the last generated, not yet in the
  // state) last; pos: how many the state holds.
  all_ = branch_.history_;
  all_.push_back(out_.tokens.back());
  position_ = static_cast<std::uint32_t>(branch_.history_.size());
  start_ = Clock::now();
  initialized_ = true;
  return {};
}

bool Llm::GenerationSession::done() const {
  return finished_ || !ran_ || out_.stopped || out_.cancelled ||
         out_.tokens.size() >= options_.max_tokens;
}

std::expected<Llm::GenerationSession::Step, std::string> Llm::GenerationSession::PrepareStep(
    bool defer_capacity) {
  base::Check(!prepared_ && !done(), "preparing a generation without a next step");
  refusal_.clear();
  if (position_ + 1 >= model_.context_) {
    ran_ = Error(
        std::format("{}'s conversation reached its context of {}", model_.name_, model_.context_));
    failed_prefix_valid_ = true;
    return std::unexpected(ran_.error());
  }
  left_ = static_cast<std::uint32_t>(options_.max_tokens - out_.tokens.size());
  certain_rows_ = 0;
  if (options_.tool_sampling) {
    options_.tool_sampling->Budget(left_, model_.context_ - position_ - 1);
    const auto cap = model_.caps_.max_segment_rows != 0 ? model_.caps_.max_segment_rows : left_;
    certain_rows_ = std::min({options_.tool_sampling->certain().size(), std::size_t{left_ - 1},
                              std::size_t{cap - 1}, std::size_t{model_.context_ - position_ - 2}});
  }
  const auto step_tokens = certain_rows_ != 0  ? static_cast<std::uint32_t>(certain_rows_ + 1)
                           : model_.speculate_ ? model_.StepTokenBoundFor(branch_, left_)
                                               : 1U;
  if (!model_.ReserveTokens(branch_, all_, all_charge_, all_.size() + step_tokens)) {
    refusal_ = "native generation history exceeds the execution budget";
    if (defer_capacity) {
      return std::unexpected(refusal_);
    }
    ran_ = Error(refusal_);
    failed_prefix_valid_ = true;
    return std::unexpected(ran_.error());
  }
  if (auto prepared = model_.PrepareDecodeStateFor(branch_, position_, left_,
                                                   static_cast<std::uint32_t>(certain_rows_));
      !prepared) {
    if (defer_capacity && model_.StateRefusedFor(branch_) && model_.StateUsableFor(branch_)) {
      // Refused before dispatch: the session stays active and unprepared,
      // its state and history as the last completed step left them.
      refusal_ = prepared.error().empty() ? "the state's capacity refused it" : prepared.error();
      return std::unexpected(refusal_);
    }
    ran_ = prepared;
    failed_prefix_valid_ = model_.StateUsableFor(branch_);
    return std::unexpected(ran_.error());
  }
  prepared_ = true;
  return Step{.all = all_,
              .position = position_,
              .left = left_,
              .speculative = model_.speculate_,
              .need_logits = options_.keep_logits || static_cast<bool>(options_.on_logits),
              .certain = certain_rows_ != 0 ? options_.tool_sampling->certain().first(certain_rows_)
                                            : std::span<const std::int32_t>{},
              .forced = options_.tool_sampling && !options_.tool_sampling->certain().empty()};
}

Status Llm::GenerationSession::FailStep(std::string error, bool prefix_valid) {
  base::Check(prepared_, "failing a generation without a prepared step");
  prepared_ = false;
  ran_ = Error(std::move(error));
  failed_prefix_valid_ = prefix_valid;
  return ran_;
}

Status Llm::GenerationSession::FailAfterAnchor(std::string error) {
  base::Check(prepared_, "failing a generation without a prepared step");
  prepared_ = false;
  ran_ = Error(std::move(error));
  if (certain_rows_ != 0 && options_.tool_sampling) {
    const auto known = options_.tool_sampling->certain().first(certain_rows_);
    all_.insert(all_.end(), known.begin(), known.end());
  }
  position_ += static_cast<std::uint32_t>(certain_rows_ + 1);
  failed_prefix_valid_ = model_.StateUsableFor(branch_);
  return ran_;
}

bool Llm::GreedyWithoutRows(const PreparedGeneration& unit) {
  return unit.branch != nullptr && unit.session != nullptr && !unit.step.need_logits &&
         !unit.branch->sampling_.has_value() && unit.branch->tool_sampling_ == nullptr;
}

Status Llm::GenerationSession::ApplyChosen(std::int32_t token) {
  base::Check(prepared_ && !model_.speculate_, "applying an unprepared ordinary generation step");
  prepared_ = false;
  return ApplyTokens({token}, {});
}

Status Llm::GenerationSession::ApplyPlain(std::vector<float> row) {
  base::Check(prepared_ && !model_.speculate_, "applying an unprepared ordinary generation step");
  prepared_ = false;
  auto next = model_.Choose(branch_, row, position_ + 1);
  if (!next) {
    ran_ = std::unexpected(next.error());
    ++position_;  // the anchor was processed before sampling failed
    failed_prefix_valid_ = model_.StateUsableFor(branch_);
    return ran_;
  }
  std::vector<std::vector<float>> logits;
  if (options_.keep_logits || options_.on_logits) {
    logits.push_back(std::move(row));
  }
  return ApplyTokens({*next}, std::move(logits));
}

Status Llm::GenerationSession::ApplyCertain(std::vector<float> row,
                                            std::vector<std::vector<float>> logits) {
  base::Check(prepared_ && options_.tool_sampling,
              "applying a certain span without prepared canonical tokens");
  auto& constraint = *options_.tool_sampling;
  auto tentative = constraint;
  const auto known = constraint.certain().first(certain_rows_);
  for (const auto token : known) {
    if (auto accepted = tentative.Accept(token); !accepted)
      return FailAfterAnchor(accepted.error());
  }
  tentative.Budget(left_ - certain_rows_, model_.context_ - position_ - certain_rows_ - 1);
  const bool more = constraint.certain().size() > certain_rows_;
  auto next = more ? std::expected<std::int32_t, std::string>(constraint.certain()[certain_rows_])
                   : model_.Choose(branch_, row, position_ + certain_rows_ + 1, &tentative);
  if (!next) return FailAfterAnchor(next.error());
  std::vector<std::int32_t> kept(known.begin(), known.end());
  kept.push_back(*next);
  // Their IDs remain in this completed unit until publication; consuming the
  // queue cannot change any borrowed engine span because all work has retired.
  constraint.ConsumeCertain(certain_rows_ + (more ? 1 : 0));
  prepared_ = false;
  return ApplyTokens(std::move(kept), std::move(logits), certain_rows_ + (more ? 1 : 0));
}

Status Llm::GenerationSession::ApplySpeculative(std::vector<std::int32_t> kept,
                                                std::vector<std::vector<float>> logits,
                                                std::uint64_t drafted) {
  base::Check(prepared_ && model_.speculate_, "applying an unprepared speculative generation step");
  prepared_ = false;
  out_.drafted += drafted;
  if (kept.empty() || kept.size() > left_ ||
      ((options_.keep_logits || options_.on_logits) && logits.size() != kept.size())) {
    // Refuse a broken runner contract before indexing borrowed rows.
    // The completed verify still retires its owed commit/restore.
    constexpr std::string_view reason =
        "a speculative step returned inconsistent token/logit counts";
    if (auto settled = model_.SettleFor(branch_); !settled) {
      ran_ = Error(std::format("{}; settling failed: {}", reason, settled.error()));
    } else {
      ran_ = Error(std::string(reason));
    }
    return ran_;
  }
  out_.accepted += kept.size() - 1;
  return ApplyTokens(std::move(kept), std::move(logits));
}

Status Llm::GenerationSession::ApplyTokens(std::vector<std::int32_t> kept,
                                           std::vector<std::vector<float>> logits,
                                           std::size_t known) {
  // User callbacks may reenter the model. Its partially published token
  // batch is not an independently completed peer until this method returns.
  struct Publishing {
    explicit Publishing(bool& flag) : active(flag) { active = true; }
    ~Publishing() { active = false; }
    bool& active;
  } publication(publishing_);
  if (++out_.steps == 1) {
    out_.first_step = Clock::now();
  }
  // The anchor and the accepted drafts are in the state now; the last
  // kept token is the next anchor. The generation ends at a stop token,
  // though the state holds what was accepted after it.
  base::Check(kept.size() <= all_.capacity() - all_.size(),
              "completed generation exceeded its funded token bound");
  position_ += static_cast<std::uint32_t>(kept.size());
  all_.insert(all_.end(), kept.begin(), kept.end());
  for (std::size_t i = 0; i < kept.size() && out_.tokens.size() < options_.max_tokens; ++i) {
    std::int32_t token = kept[i];
    if (options_.tool_sampling && i + 1 == kept.size() &&
        options_.tool_sampling->certain().empty()) {
      options_.tool_sampling->Budget(options_.max_tokens - out_.tokens.size(),
                                     model_.context_ - position_);
      auto planned = options_.tool_sampling->Plan(token);
      if (!planned) {
        ran_ = std::unexpected(planned.error());
        out_.cancelled = true;
        break;
      }
      token = planned->tokens.front();
      all_.back() = token;  // the local, not-yet-processed anchor only
      options_.tool_sampling->Certain({planned->tokens.begin() + 1, planned->tokens.end()});
    }
    if (options_.tool_sampling) {
      if (auto accepted = options_.tool_sampling->Accept(token); !accepted) {
        ran_ = std::unexpected(accepted.error());
        out_.cancelled = true;
        break;
      }
    }
    out_.tokens.push_back(token);
    if (options_.keep_logits) {
      out_.logits.push_back(logits[i]);
    }
    if (options_.on_logits && !options_.on_logits(token, logits[i])) {
      out_.cancelled = true;
    }
    if (IsStop(token)) {
      out_.stopped = true;
    }
    // A scoring response associates each visible token with its row.
    // Deliver it now, so a stop string cannot collect later verify rows.
    // The entire accepted verify is already complete and is still settled.
    if (options_.on_logits || options_.tool_sampling) {
      out_.cancelled = !Report() || out_.cancelled;
    }
    if (out_.stopped || out_.cancelled) {
      if (out_.cancelled && !out_.stopped && options_.tool_sampling && i + 1 < known) {
        // A callback can yield within this completed unit. Only delivered IDs
        // advanced the lexical cursor; preserve its still-known suffix for the
        // restored public prefix, rather than resampling or losing the grammar.
        std::vector<std::int32_t> pending(kept.begin() + static_cast<std::ptrdiff_t>(i + 1),
                                          kept.begin() + static_cast<std::ptrdiff_t>(known));
        const auto tail = options_.tool_sampling->certain();
        pending.insert(pending.end(), tail.begin(), tail.end());
        options_.tool_sampling->Certain(std::move(pending));
      }
      break;
    }
  }
  if (!options_.on_logits && !options_.tool_sampling) {
    out_.cancelled = !Report() || out_.cancelled;
  }
  return {};
}

void Llm::GenerationSession::EndRefused() {
  base::Check(refused() && !prepared_ && ran_.has_value(),
              "ending a generation that was not refused");
  // Named as a prefill's refusal names its chunk (positions only, D-014).
  ran_ = Error(std::format("{}'s decode step at {}: {}", model_.name_, position_,
                           std::exchange(refusal_, {})));
  failed_prefix_valid_ = model_.StateUsableFor(branch_);
}

Status Llm::GenerationSession::RunScalarStep(bool defer_capacity) {
  auto step = PrepareStep(defer_capacity);
  if (!step) {
    return std::unexpected(step.error());
  }
  const auto started = Clock::now();
  if (step->speculative && !step->forced) {
    std::vector<std::int32_t> kept;
    std::vector<std::vector<float>> logits;
    bool prefix_kept = false;
    auto ran =
        model_.SpecStepFor(branch_, step->all, step->position, step->left, kept,
                           step->need_logits ? &logits : nullptr, out_.drafted, &prefix_kept);
    if (!ran) {
      return FailStep(ran.error(), prefix_kept);
    }
    // Its decode speed (calibration.h: the decode floor's).
    model_.calibration_samples_.DecodeStep(step->position, static_cast<double>(kept.size()),
                                           Seconds(Clock::now() - started));
    // SpecStep already updated the legacy counter, including partial failures.
    return ApplySpeculative(std::move(kept), std::move(logits), 0);
  }
  PreparedGeneration unit{.session = this,
                          .branch = &branch_,
                          .step = *step,
                          .row = {},
                          .kept = {},
                          .logits = {},
                          .drafted = 0,
                          .result = {},
                          .failed_prefix_valid = false,
                          .anchor_processed = false,
                          .chosen = std::nullopt};
  model_.RunPlain(std::span(&unit, 1));
  if (!unit.result) {
    return unit.anchor_processed ? FailAfterAnchor(unit.result.error())
                                 : FailStep(unit.result.error(), unit.failed_prefix_valid);
  }
  model_.calibration_samples_.DecodeStep(
      step->position, static_cast<double>(step->forced ? step->certain.size() + 1 : 1),
      Seconds(Clock::now() - started));
  if (unit.step.forced) return ApplyCertain(std::move(unit.row), std::move(unit.logits));
  return unit.chosen ? ApplyChosen(*unit.chosen) : ApplyPlain(std::move(unit.row));
}

Status Llm::RunGenerationWave(std::span<GenerationSession* const> sessions, bool defer_capacity) {
  if (sessions.empty() || sessions.size() > kMaxBranches ||
      sessions.size() > generation_wave_capacity() ||
      (sessions.size() > 1 && !supports_generation_waves())) {
    return Error("the generation cohort exceeds this model's capability");
  }
  std::array<bool, kMaxBranches> selected{};
  for (GenerationSession* session : sessions) {
    if (session == nullptr || &session->model_ != this || session->done() || session->prepared_ ||
        session->finished_) {
      return Error("a generation cohort contains an inactive or foreign session");
    }
    const std::size_t slot = BranchIndex(session->branch_);
    if (selected[slot]) {
      return Error("a generation cohort repeats a native branch");
    }
    selected[slot] = true;
  }
  std::vector<PreparedGeneration> prepared;
  prepared.reserve(sessions.size());
  const auto fail_cohort = [&](const std::string& error) {
    for (GenerationSession* session : sessions) {
      if (session->prepared_) {
        [[maybe_unused]] const auto failed = session->FailStep(error);
      } else {
        session->ran_ = Error(error);
        session->refusal_.clear();  // ended, no longer retryable
        session->failed_prefix_valid_ = false;
      }
    }
  };
  for (GenerationSession* session : sessions) {
    auto step = session->PrepareStep(defer_capacity);
    if (!step) {
      // PrepareStep records this session's own terminal error, or with
      // defer_capacity its retryable refusal. A refusal confined to its
      // branch does not discard another prepared step.
      if (!GenerationCohortUsable()) {
        fail_cohort(step.error());
        return std::unexpected(step.error());
      }
      continue;
    }
    prepared.push_back({.session = session,
                        .branch = &session->branch_,
                        .step = *step,
                        .row = {},
                        .kept = {},
                        .logits = {},
                        .drafted = 0,
                        .result = {},
                        .failed_prefix_valid = false,
                        .anchor_processed = false,
                        .chosen = std::nullopt});
  }
  if (prepared.empty()) {
    return {};  // every session retains its own preparation error
  }
  std::ranges::sort(prepared, {},
                    [this](const PreparedGeneration& unit) { return BranchIndex(*unit.branch); });
  const auto started = Clock::now();
  if (auto ran = RunPreparedGenerationWave(prepared); !ran) {
    for (PreparedGeneration& unit : prepared) {
      unit.session->out_.drafted += unit.drafted;
    }
    fail_cohort(ran.error());
    return ran;
  }
  {
    // Each request's decode speed in the wave (calibration.h: the decode
    // floor's): the tokens it committed, on average, over the wave's time.
    const double seconds = Seconds(Clock::now() - started);
    std::size_t tokens = 0;
    std::uint32_t at = std::numeric_limits<std::uint32_t>::max();
    for (const PreparedGeneration& unit : prepared) {
      if (unit.result) {
        tokens += unit.step.forced        ? unit.step.certain.size() + 1
                  : unit.step.speculative ? unit.kept.size()
                                          : 1;
      }
      at = std::min(at, unit.step.position);
    }
    calibration_samples_.DecodeStep(
        at, static_cast<double>(tokens) / static_cast<double>(prepared.size()), seconds);
  }
  for (PreparedGeneration& unit : prepared) {
    if (!unit.result) {
      unit.session->out_.drafted += unit.drafted;
      [[maybe_unused]] const auto failed =
          unit.anchor_processed
              ? unit.session->FailAfterAnchor(unit.result.error())
              : unit.session->FailStep(unit.result.error(), unit.failed_prefix_valid);
    } else {
      // Apply owns any ordinary sampling/callback error on this session.
      // Completed peers still receive their results in branch order.
      (void)(unit.step.forced
                 ? unit.session->ApplyCertain(std::move(unit.row), std::move(unit.logits))
             : unit.step.speculative
                 ? unit.session->ApplySpeculative(std::move(unit.kept), std::move(unit.logits),
                                                  unit.drafted)
             : unit.chosen ? unit.session->ApplyChosen(*unit.chosen)
                           : unit.session->ApplyPlain(std::move(unit.row)));
    }
    if (!GenerationCohortUsable()) {
      const std::string error = "the native generation cohort became unusable";
      fail_cohort(error);
      return Error(error);
    }
  }
  return {};
}

void Llm::GenerationSession::Cancel() {
  base::Check(!finished_ && !prepared_, "cancelling a generation away from a completed boundary");
  out_.cancelled = true;
}

void Llm::GenerationSession::Close() {
  base::Check(branch_.generation_session_ == this, "a generation lost its branch ownership");
  branch_.generation_session_ = nullptr;
  branch_.sampling_.reset();
  branch_.tool_sampling_ = nullptr;
  std::vector<float>().swap(branch_.constrained_logits_);
  model_.RetireSamplingScratchFor(branch_);
  model_.ReleaseTokens(all_, all_charge_);
  branch_.generation_active_ = false;
  finished_ = true;
}

Status Llm::GenerationSession::Finish() {
  if (finished_) {
    return ran_;
  }
  base::Check(!prepared_, "finishing a generation before its prepared step was applied");
  base::Check(done(), "finishing an active generation without cancellation");
  out_.decode_seconds = Seconds(Clock::now() - start_);
  branch_.sampling_.reset();
  if (!ran_) {
    if (failed_prefix_valid_) {
      if (auto settled = model_.SettleFor(branch_); !settled) {
        ran_ = Error(std::format("{}; settling failed: {}", ran_.error(), settled.error()));
      } else {
        model_.PublishTokens(branch_, all_, all_charge_, position_);
        branch_.needs_clear_ = false;
        Close();
        return ran_;
      }
    }
    branch_.needs_clear_ = true;
    model_.ReleaseTokens(branch_.history_, branch_.history_charge_);
    Close();
    return ran_;
  }
  model_.PublishTokens(branch_, all_, all_charge_, position_);
  branch_.history_used_ = Clock::now();
  if (out_.tokens.size() > options_.max_tokens) {
    out_.tokens.resize(options_.max_tokens);
  }
  if (out_.logits.size() > out_.tokens.size()) {
    out_.logits.resize(out_.tokens.size());
  }
  if (auto settled = model_.SettleFor(branch_); !settled) {
    branch_.needs_clear_ = true;
    model_.ReleaseTokens(branch_.history_, branch_.history_charge_);
    ran_ = settled;
  }
  Close();
  return ran_;
}

Status Llm::Generate(Branch& branch, const std::vector<float>& last, const GenerateOptions& options,
                     Generation& out, bool resume) {
  CheckBranch(branch);
  branch.capacity_refused_ = false;
  auto began = BeginGeneration(branch, last, options, out, resume);
  if (!began) {
    return std::unexpected(began.error());
  }
  auto& session = **began;
  std::size_t reclaimed = 0;
  while (!session.done()) {
    if (auto ran = session.RunScalarStep(true); ran) {
      reclaimed = 0;
      continue;
    }
    if (session.refused()) {
      // The same step again once capacity was freed; otherwise it ends at
      // its completed prefix, as an undeferred refusal does.
      if (reclaimed < kMaxBranches && ReclaimFor(branch)) {
        ++reclaimed;
        continue;
      }
      branch.capacity_refused_ = true;
      session.EndRefused();
    }
    break;
  }
  auto finished = session.Finish();
  // Settling the refused generation can still fail; that failure (the
  // branch then owes a clear) is not capacity's.
  branch.capacity_refused_ = branch.capacity_refused_ && !branch.needs_clear_;
  return finished;
}

}  // namespace llmp::runtime
