// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "runtime/service_backend.h"

namespace llmp::runtime::service_detail {
bool NodeBackend::Supports(const api::CooperativeBackend::Request& request) const {
  Served* model = server_.Find(request.options.model);
  return model != nullptr && model->llm() && static_cast<Llm&>(*model).supports_generation_waves();
}

std::expected<std::unique_ptr<api::CooperativeBackend::Work>, api::Error> NodeBackend::Start(
    const api::CooperativeBackend::Request& request, api::Exchange& exchange) {
  if (!Supports(request)) {
    return std::unique_ptr<api::CooperativeBackend::Work>{};
  }
  auto& model = static_cast<Llm&>(*server_.Find(request.options.model));
  if (cohort_model_ != nullptr && cohort_model_ != &model) {
    return std::unique_ptr<api::CooperativeBackend::Work>{};
  }
  const auto claimed =
      std::ranges::count_if(cohort_, [](const ChatWork* frame) { return frame != nullptr; });
  if (static_cast<std::size_t>(claimed) >= model.generation_wave_capacity() ||
      std::ranges::find(cohort_, nullptr) == cohort_.end()) {
    return std::unique_ptr<api::CooperativeBackend::Work>{};
  }
  // Members waiting for state capacity come first (FIFO): a new prompt
  // would compete for the capacity they wait for.
  if (!AdmissionOpen(Snapshot())) {
    return std::unique_ptr<api::CooperativeBackend::Work>{};
  }
  // A request already found no memory for a slot waits for a peer to
  // retire before it is looked at again (below).
  if (memory_wait_.Blocked(static_cast<std::size_t>(claimed))) {
    return std::unique_ptr<api::CooperativeBackend::Work>{};
  }
  // A request that yielded its place continues from its generation so far
  // (ChatResume); any other is rendered now.
  const auto* resume = static_cast<const ChatResume*>(request.resume);
  std::shared_ptr<api::LiteralOutput> literal = resume != nullptr ? resume->literal : nullptr;
  std::expected<ChatPrompt, api::Error> rendered;
  if (resume != nullptr) {
    rendered = resume->rendered;
  } else if (request.literal != nullptr) {
    auto prepared =
        api::PrepareLiteralPrompt(*request.literal, model.tokenizer(), model.usable_context(),
                                  memory_->capacity(), memory_.get());
    if (!prepared) {
      return std::unexpected(prepared.error());
    }
    auto output =
        api::LiteralOutput::Create(*request.literal, *prepared, model.tokenizer(), *memory_);
    if (!output) {
      return std::unexpected(output.error());
    }
    literal = std::move(*output);
    auto held = api::PromptTokens::Hold(std::move(prepared->tokens), *memory_);
    if (!held) {
      return std::unexpected(held.error());
    }
    rendered = ChatPrompt{.token_storage = std::move(*held),
                          .max_tokens = request.options.max_tokens.value_or(16)};
  } else {
    rendered = PrepareChat(model, request.options, exchange, *memory_);
  }
  if (!rendered) {
    return std::unexpected(rendered.error());
  }
  const std::vector<std::int32_t>& wanted =
      resume != nullptr ? resume->history : rendered->tokens();
  // Choose only an unclaimed model-owned branch. Prefix matching is a
  // reuse preference, never conversation identity or retention policy:
  // the branch whose state this prompt can actually reuse the most of
  // (Llm::ReusablePrefix: its live history continued, or a turn
  // checkpoint inside the common prefix); with none, an empty branch,
  // then the least recently used, so a new conversation does not
  // discard an idle one's state while an empty branch is free.
  std::size_t slot = cohort_.size();
  std::size_t best = 0;
  bool best_empty = false;
  Clock::time_point best_used;
  for (std::size_t i = 0; i < model.branches() && i < cohort_.size(); ++i) {
    if (cohort_[i] != nullptr) {
      continue;
    }
    auto branch = model.branch(i);
    if (!branch) {
      return std::unexpected(Failure(500, "the model's conversation branch is unavailable"));
    }
    if (resume != nullptr && resume->held_branch != nullptr && *branch != resume->held_branch) {
      continue;  // logical identity survives the swap; prefix matching is insufficient
    }
    if ((*branch)->HeldByContinuation() && (resume == nullptr || resume->held_branch != *branch)) {
      continue;
    }
    const std::size_t reusable = literal != nullptr ? 0 : model.ReusablePrefix(**branch, wanted);
    const bool empty = !(*branch)->HasRetainedState();
    const Clock::time_point used = model.LastUsed(**branch);
    const bool better =
        slot == cohort_.size() || reusable > best ||
        (reusable == best && ((empty && !best_empty) || (empty == best_empty && used < best_used)));
    if (better) {
      slot = i;
      best = reusable;
      best_empty = empty;
      best_used = used;
    }
  }
  if (slot == cohort_.size()) {
    return std::unique_ptr<api::CooperativeBackend::Work>{};
  }
  auto branch = model.branch(slot);
  if (!branch) {
    return std::unexpected(Failure(500, "the model's conversation branch is unavailable"));
  }
  // Request slots follow memory (docs/runtime-serving.md#request-slots):
  // beside its peers, a request joins only where the budget holds its
  // prompt's state and what its peers' prompts have yet to take, free or
  // freed through the reclaim order (idle conversations spilled, stale
  // plans and graphs dropped). Otherwise it waits first in the queue until
  // a peer retires. A lone request always starts: the capacity policy
  // governs it as before.
  if (claimed != 0 && server_.resident() == &model) {
    // An estimate that failed is no estimate of nothing: the request is
    // refused, not admitted for room the budget may lack.
    auto needed = PromptStateToCome(model, **branch, rendered->tokens().size());
    for (const ChatWork* peer : cohort_) {
      if (needed && peer != nullptr && !peer->terminal() &&
          peer->stage == ChatWork::Stage::kPrompt) {
        auto more = PromptStateToCome(model, peer->branch, peer->rendered.tokens().size());
        needed = more ? std::expected<std::uint64_t, std::string>(*needed + *more) : more;
      }
    }
    if (!needed) {
      return std::unexpected(Failure(
          500, "the request's state could not be measured for a request slot: " + needed.error()));
    }
    if (*needed != 0 && !server_.RoomFor(*needed, "a request slot", *branch)) {
      memory_wait_.NoRoom();
      Say(log_, std::format("{}'s request waits for memory for a request slot: {:.1f} MiB of "
                            "prompt state to come for {} tokens beside {} active requests",
                            model.name(), static_cast<double>(*needed) / (1U << 20U),
                            rendered->tokens().size(), claimed));
      return std::unique_ptr<api::CooperativeBackend::Work>{};
    }
  }
  Floors floors = model.floors();
  if (literal != nullptr && literal->score_prompt()) {
    floors.prefill = floors.decode;  // teacher forcing is one-row target work
  }
  const auto swap_bytes = SwapBytes(model);
  auto frame =
      std::make_unique<ChatWork>(*this, model, **branch, slot, request.options, exchange,
                                 std::move(*rendered), floors, swap_bytes, std::move(literal));
  if (resume != nullptr) {
    // Its state rebuilt to the history it yielded at (restored, reused or
    // prefilled), then its generation continues from the anchor
    // (BeginChatGeneration's resume): nothing is chosen or sent again.
    frame->resume = !resume->generation.tokens.empty();
    frame->continued = true;
    frame->switch_restore = resume->held_branch != nullptr;
    frame->resume_tokens = resume->history;
    frame->generation = resume->generation;
    frame->options.sampling = resume->sampling;
    frame->options.seed = resume->seed;
    frame->output.Restore(resume->output);
    frame->reused = resume->reused;
    if (!frame->FundTokens(false)) {
      frame->retired = true;  // no session or native work borrowed it
      return std::unexpected(*frame->error);
    }
  }
  auto prompt = WorkPrompt(*frame);
  if (!prompt) {
    frame->retired = true;  // no session or native work retained a reference
    return std::unexpected(Failure(500, "the prompt could not be admitted: " + prompt.error()));
  }
  frame->prompt_session = std::move(*prompt);
  if (!exchange.Admit({.prompt_tokens = static_cast<std::uint32_t>(frame->rendered.tokens().size()),
                       .max_tokens = frame->rendered.max_tokens,
                       .swap_bytes = swap_bytes,
                       .floors = floors})) {
    frame->Cancel();  // retired normally; no native unit has run
  }
  frame->admitted = ++admissions_;
  cohort_model_ = &model;
  cohort_[slot] = frame.get();
  return std::unique_ptr<api::CooperativeBackend::Work>(std::move(frame));
}

std::expected<api::CooperativeBackend::Unit, std::string> NodeBackend::NextUnit(
    std::span<api::CooperativeBackend::Work* const> work) {
  if (auto valid = CheckCohort(work); !valid) {
    return std::unexpected(valid.error());
  }
  selected_prompt_ = nullptr;
  selected_prefill_count_ = 0;
  selected_decode_.clear();
  selected_swap_ = server_.resident() != cohort_model_;
  if (selected_swap_) {
    return api::CooperativeBackend::Unit{
        .phase = Phase::kSwap,
        .expected_seconds =
            static_cast<double>(SwapBytes(*cohort_model_)) / kSwapFloorBytesPerSecond};
  }
  // The schedule (cohort_schedule.h): the prompt with the fewest tokens
  // left goes next, a long one passed over for a bounded number of units,
  // and a generating member waits for at most one prompt unit.
  std::vector<ScheduledMember> members;
  std::vector<ChatWork*> frames;
  bool paused = false;
  const bool restoring_generations = std::ranges::any_of(work, [](const auto* base) {
    const auto& frame = static_cast<const ChatWork&>(*base);
    return frame.switch_restore && frame.resume && frame.stage == ChatWork::Stage::kPrompt &&
           frame.wait == CapacityWait::kNone;
  });
  for (auto* base : work) {
    auto& frame = static_cast<ChatWork&>(*base);
    if (frame.wait != CapacityWait::kNone) {
      continue;  // waits for state capacity (cohort_capacity.h)
    }
    if (frame.stage == ChatWork::Stage::kGeneration) {
      if (restoring_generations) {
        continue;  // restore the old wave before any member decodes at a narrower width
      }
      // A stream whose client is behind waits at its completed step
      // (backpressure, D-102); the others go on.
      if (frame.exchange.Paused()) {
        paused = true;
        continue;
      }
      selected_decode_.push_back(&frame);
      members.push_back({.stage = ScheduledMember::Stage::kGeneration,
                         .admitted = frame.admitted,
                         .remaining = 0,
                         .passed = 0,
                         .waited = frame.waited,
                         .finishing = false});
      frames.push_back(&frame);
    } else if (frame.stage == ChatWork::Stage::kPrompt) {
      if (restoring_generations && !(frame.switch_restore && frame.resume)) {
        continue;
      }
      const std::uint32_t remaining = frame.prompt_session->remaining_rows();
      members.push_back(
          {.stage = ScheduledMember::Stage::kPrompt,
           .admitted = frame.admitted,
           .remaining = remaining,
           .passed = frame.passed,
           .waited = 0,
           .finishing = frame.prompt_session->started() && remaining <= kPromptFinishRows});
      frames.push_back(&frame);
    }
  }
  const ScheduleChoice choice = NextCohortUnit(members);
  ChatWork* prompt = choice.prompt ? frames[*choice.prompt] : nullptr;
  if (choice.decode) {
    double expected = 0;
    for (const ChatWork* frame : selected_decode_) {
      expected +=
          ExpectedSeconds(Phase::kDecode, frame->model.speculative() ? 4 : 1, frame->floors);
    }
    return api::CooperativeBackend::Unit{.phase = Phase::kDecode, .expected_seconds = expected};
  }
  if (prompt == nullptr || !cohort_model_->supports_causal_waves()) selected_decode_.clear();
  if (prompt == nullptr && paused) {
    // Every runnable member waits for its client: the server waits for
    // one to read or leave.
    return api::CooperativeBackend::Unit{.phase = Phase::kPaused, .expected_seconds = 0};
  }
  if (prompt == nullptr) {
    return std::unexpected("an active chat cohort has no next completed unit");
  }
  std::array<Llm::PromptSession*, Llm::kMaxBranches> prompt_sessions{};
  for (std::size_t i = 0; i < frames.size(); ++i) {
    if (frames[i]->stage == ChatWork::Stage::kPrompt)
      prompt_sessions[i] = frames[i]->prompt_session.get();
  }
  const auto ready = Llm::PromptSession::SelectForWave(
      choice.decode, choice.prompt, std::span(prompt_sessions).first(frames.size()));
  base::Check(ready.has_value(), "a selected prompt has no readiness unit");
  prompt = frames[*ready];
  auto unit = prompt->prompt_session->NextUnit();
  if (!unit) {
    return std::unexpected(unit.error());
  }
  selected_prompt_ = prompt;
  selected_prefill_[0] = prompt;
  selected_prefill_count_ = 1;
  double expected = ExpectedSeconds(Phase::kPrefill, unit->rows, prompt->floors);
  // Keep the scheduler's shortest/aged choice first. Only an already
  // runnable compatible chunk may share this same completed prompt unit.
  for (ChatWork* peer : frames) {
    if (selected_prefill_count_ >= cohort_model_->prefill_wave_capacity()) break;
    if (peer == prompt || peer->stage != ChatWork::Stage::kPrompt ||
        !prompt->prompt_session->CanJoin(*peer->prompt_session))
      continue;
    const auto other = peer->prompt_session->NextUnit();
    base::Check(other.has_value(), "a compatible prompt has no declared unit");
    selected_prefill_[selected_prefill_count_++] = peer;
    expected += ExpectedSeconds(Phase::kPrefill, other->rows, peer->floors);
  }
  if (unit->phase != Llm::PromptSession::Phase::kChunk) {
    selected_decode_.clear();
    expected +=
        static_cast<double>(prompt->branch.state_snapshot_bytes()) / kSwapFloorBytesPerSecond;
  }
  const auto room = cohort_model_->generation_wave_capacity() > selected_prefill_count_
                        ? cohort_model_->generation_wave_capacity() - selected_prefill_count_
                        : 0;
  if (selected_decode_.size() > room) selected_decode_.resize(room);
  for (const auto* frame : selected_decode_)
    expected += ExpectedSeconds(Phase::kDecode, 1, frame->floors);
  return api::CooperativeBackend::Unit{.phase = Phase::kPrefill, .expected_seconds = expected};
}

std::expected<void, std::string> NodeBackend::Advance(
    std::span<api::CooperativeBackend::Work* const> work) {
  if (auto valid = CheckCohort(work); !valid) {
    return valid;
  }
  // Whether this unit's waits were cancelled for a hang (rung 1), asked
  // after it.
  (void)server_.node().TakeHangCancelled();
  if (selected_swap_) {
    selected_swap_ = false;
    bool needed = false;
    for (auto* base : work) {
      auto& frame = static_cast<ChatWork&>(*base);
      if (!frame.exchange.Next(Phase::kSwap, frame.swap_bytes)) {
        frame.Cancel();
      } else {
        needed = true;
      }
    }
    if (!needed) {
      return {};
    }
    swapped_ = server_.resident() != cohort_model_;
    if (auto activated = server_.Activate(*cohort_model_, parts_); !activated) {
      swapped_ = false;
      if (server_.node().TakeHangCancelled()) {
        Say(log_,
            "hang recovery: the swap's hung work was cancelled; the swap failed and only "
            "the requests that needed it fail");
        (void)AwaitDrained();  // a read still in the drive: rung 3
      }
      if (server_.NodeHealthy()) {
        // Recovery first (D-102): a swap refused for room, or failed and
        // undone (or not: no model resident, the next activation loads one
        // whole), fails only these requests. The cohort posted no native
        // work, so its retirement has no references to fence.
        for (auto* base : work) {
          auto& frame = static_cast<ChatWork&>(*base);
          frame.error = Failure(503, "the model could not be made resident: " + activated.error());
        }
        return {};
      }
      Fail("making the cooperative chat model resident: " + activated.error());
      return activated;
    }
    // Resident: the cohort's native work begins, whose references its
    // retirement fences.
    cohort_native_touched_ = true;
    for (auto* base : work) {
      static_cast<ChatWork&>(*base).native_touched = true;
    }
    return {};
  }
  if (selected_prompt_ == nullptr && selected_decode_.empty()) {
    return std::unexpected("advancing chat without its declared unit");
  }
  // Selection may itself change native ownership before it reports an
  // error. Retirement therefore needs the fence even after refusal here.
  cohort_native_touched_ = true;
  // All active states stay protected even while just one branch prefills.
  if (auto selected = SelectCohort(); !selected) {
    if (server_.node().TakeHangCancelled()) {
      return HangCohort("selecting the native chat cohort");
    }
    Fail("selecting the native chat cohort: " + selected.error());
    return selected;
  }
  if (selected_prompt_ != nullptr && (selected_prefill_count_ > 1 || !selected_decode_.empty()))
    return AdvancePrefillWave();
  if (selected_prompt_ != nullptr) {
    selected_prefill_count_ = 0;
    ChatWork& frame = *std::exchange(selected_prompt_, nullptr);
    frame.native_touched = true;
    // The schedule's counters (cohort_schedule.h), as this unit runs; a
    // unit that prefills no rows (a reuse, a checkpoint) ages no one.
    const auto unit = frame.prompt_session->NextUnit();
    const bool ages = unit && unit->rows > 0;
    for (ChatWork* other : cohort_) {
      if (other == nullptr || other == &frame) {
        continue;
      }
      if (other->stage == ChatWork::Stage::kGeneration) {
        ++other->waited;
      } else if (other->stage == ChatWork::Stage::kPrompt && ages) {
        ++other->passed;
      }
    }
    frame.passed = 0;
    const PrefillGoOn go_on = [&frame](std::uint32_t rows) {
      return !frame.cancelled && frame.exchange.Next(Phase::kPrefill, rows);
    };
    if (auto advanced = frame.prompt_session->Advance(go_on, true); !advanced) {
      if (server_.node().TakeHangCancelled()) {
        return HangCohort("a prompt's unit");
      }
      std::string error = "the prompt could not be processed: " + advanced.error();
      if (frame.prompt_session->refused()) {
        frame.refusal = std::move(error);
        const std::array<ChatWork*, 1> refused = {&frame};
        WaitForCapacity(refused);
      } else {
        frame.error = Failure(500, std::move(error));
      }
      Rebalance();
      return {};  // own failure; Retire independently proves references
    }
    if (server_.node().TakeHangCancelled()) {
      return HangCohort("a prompt's unit");  // a step ended after its request was cancelled
    }
    if (frame.literal != nullptr && unit && unit->phase == Llm::PromptSession::Phase::kReuse) {
      frame.literal->PromptStarted();
    }
    if (frame.prompt_session->done()) {
      auto begun = BeginChatGeneration(frame);
      Rebalance();
      return begun;
    }
    return {};
  }
  std::vector<Llm::GenerationSession*> sessions;
  std::vector<ChatWork*> stepped;
  sessions.reserve(selected_decode_.size());
  stepped.reserve(selected_decode_.size());
  for (ChatWork* frame : selected_decode_) {
    if (!frame->exchange.Next(Phase::kDecode, 0)) {
      frame->Cancel();
      continue;
    }
    frame->native_touched = true;
    sessions.push_back(frame->generation_session.get());
    stepped.push_back(frame);
  }
  selected_decode_.clear();
  if (sessions.empty()) {
    return {};
  }
  if (auto advanced = cohort_model_->RunGenerationWave(sessions, true); !advanced) {
    if (server_.node().TakeHangCancelled()) {
      return HangCohort("a generation wave");
    }
    Fail("the native chat generation wave failed: " + advanced.error());
    return advanced;
  }
  if (server_.node().TakeHangCancelled()) {
    return HangCohort("a generation wave");  // a step ended after its request was cancelled
  }
  for (ChatWork* frame : stepped) {
    frame->waited = 0;
  }
  std::vector<ChatWork*> refused;
  for (ChatWork* frame : stepped) {
    if (frame->generation_session->refused()) {
      frame->refusal = "the generation failed: " + frame->generation_session->refusal();
      refused.push_back(frame);
    }
  }
  if (!refused.empty()) {
    WaitForCapacity(refused);
  }
  Rebalance();
  return {};
}

api::CooperativeBackend::Retirement NodeBackend::Retire(api::CooperativeBackend::Work& work) {
  auto& frame = static_cast<ChatWork&>(work);
  if (&frame.owner != this || frame.retired || frame.slot >= cohort_.size() ||
      cohort_[frame.slot] != &frame || !frame.terminal()) {
    return {.result = std::unexpected(Failure(500, "retiring an invalid chat owner")),
            .references_retired = false};
  }
  if (frame.prompt_session != nullptr) {
    if (!frame.prompt_session->done()) {
      frame.prompt_session->Cancel();
    }
    if (!frame.resume && !frame.continued) {
      frame.reused = frame.prompt_session->reused();  // a rebuild's reuse is not the turn's
    }
    if (auto finished = frame.prompt_session->Finish(); !finished && !frame.error) {
      frame.error = Failure(500, "the prompt failed: " + finished.error());
    }
  }
  if (frame.generation_session != nullptr) {
    if (!frame.generation_session->done()) {
      frame.generation_session->Cancel();
    }
    if (auto finished = frame.generation_session->Finish(); !finished && !frame.error) {
      frame.error = Failure(500, "the generation failed: " + finished.error());
    }
  }
  const auto peers =
      std::ranges::count_if(cohort_, [](const ChatWork* peer) { return peer != nullptr; });
  if (cohort_native_touched_ && hang_reset_ != nullptr) {
    // After a hang's cancellation (rung 2) the cohort may be faulted: a
    // fence of the model's stream, not the cohort's retirement, proves
    // that nothing queued still borrows this owner's frame. A fence that
    // cannot be made, or a cancellation that never drained, leaves
    // nothing less than a restart (rung 3).
    if (!AwaitDrained()) {
      return {.result = std::unexpected(HungFailure()), .references_retired = false};
    }
    if (auto fenced = server_.FenceModel(frame.model); !fenced) {
      if (ladder_ != nullptr) {
        ladder_->Restart(
            std::format("the {} cohort's stream could not be fenced after its "
                        "hung work was cancelled: {}",
                        frame.model.name(), fenced.error()));
      }
      return {.result = std::unexpected(HungFailure()), .references_retired = false};
    }
  } else if (cohort_native_touched_) {
    // The explicit fence hook, not a session status or StateUsable flag,
    // proves that native copies/jobs no longer borrow this owner's frame.
    const auto retired = server_.RetireRequestBranches(frame.model, peers == 1);
    if (!retired.references_retired) {
      return {.result = std::unexpected(Failure(500, "native chat references did not retire")),
              .references_retired = false};
    }
    if (!retired.result) {
      Fail("ending the native chat cohort: " + retired.result.error());
      if (!frame.error) {
        frame.error = Failure(500, "the native chat cohort failed during retirement");
      }
    }
  }
  // Yielding its place (D-102): its state stays as a finished turn's, and
  // it continues from its generation so far when taken up again, unless
  // the generation ended whole anyway.
  const bool yields = frame.yielding && !frame.error && !frame.generation.stopped &&
                      ((frame.model_paused && frame.generation.tokens.empty()) ||
                       (!frame.generation.tokens.empty() &&
                        frame.generation.tokens.size() < frame.options.max_tokens));
  std::shared_ptr<api::Yielded> yielded;
  if (yields) {
    if (frame.model_paused) {
      Say(log_, std::format("{}'s request paused for a model switch: {} of {} prompt tokens "
                            "processed, {} generated",
                            frame.model.name(),
                            std::min(frame.branch.history().size(), frame.rendered.tokens().size()),
                            frame.rendered.tokens().size(), frame.generation.tokens.size()));
    }
    auto kept = KeepContinuation(frame.rendered, frame.generation, frame.options,
                                 frame.output.Save(), frame.reused, *memory_,
                                 frame.model_paused ? &frame.branch : nullptr, frame.literal);
    if (!kept) {
      frame.error = kept.error();
    } else {
      yielded = std::move(*kept);
    }
  } else if (!frame.error && !frame.generation.tokens.empty()) {
    if (frame.literal == nullptr) {
      frame.output.Finish();  // including a generation preempted and not yet resumed
    }
  }
  api::LiteralResult literal_result;
  if (frame.literal != nullptr && !yields) {
    if (!frame.error) {
      (void)frame.literal->Finish(
          [&frame](std::string_view piece) { return frame.exchange.Content(piece); });
      if (frame.literal->error()) {
        frame.error = *frame.literal->error();
      }
    }
    literal_result = frame.literal->TakeResult();
  }
  if (!frame.error && ladder_ != nullptr) {
    ladder_->Served(frame.model.name());  // a later hang may reset it again
  }
  api::Completion result{
      .completion_tokens = static_cast<std::uint32_t>(frame.generation.tokens.size()),
      .cached_tokens = frame.literal != nullptr ? 0 : frame.reused,
      .stopped = frame.rendered.max_tokens == 0 || frame.generation.stopped,
      .literal = std::move(literal_result),
      .yielded = std::move(yielded)};
  cohort_[frame.slot] = nullptr;
  frame.retired = true;
  memory_wait_.MemberRetired();  // its state is idle now, and reclaimable
  frame.generation_session.reset();
  frame.prompt_session.reset();
  if (peers == 1) {
    cohort_model_ = nullptr;
    cohort_native_touched_ = false;
    selected_prompt_ = nullptr;
    selected_prefill_count_ = 0;
    selected_decode_.clear();
    if (Llm* hung = std::exchange(hang_reset_, nullptr); hung != nullptr) {
      // The last member of a cohort whose work hung: the model is reset.
      (void)RecoverFromHang(*hung, std::format("the {} cohort's unit", hung->name()));
    }
  } else {
    // Its state is no longer leased once the cohort is next selected:
    // members waiting for capacity retry (cohort_capacity.h).
    Cohort cohort = Snapshot();
    OnMemberRetired(cohort);
    Apply(cohort);
    Rebalance();
  }
  return {.result = frame.error
                        ? std::expected<api::Completion, api::Error>(std::unexpected(*frame.error))
                        : std::expected<api::Completion, api::Error>(std::move(result)),
          .references_retired = true};
}

Status NodeBackend::AdvancePrefillWave() {
  const auto selected = std::span(selected_prefill_).first(selected_prefill_count_);
  selected_prompt_ = nullptr;
  selected_prefill_count_ = 0;
  std::array<Llm::PromptSession*, Llm::kMaxBranches> sessions{};
  std::array<PrefillGoOn, Llm::kMaxBranches> callbacks{};
  std::array<std::uint32_t, Llm::kMaxBranches> before{};
  for (std::size_t i = 0; i < selected.size(); ++i) {
    ChatWork* frame = selected[i];
    frame->native_touched = true;
    sessions[i] = frame->prompt_session.get();
    before[i] = sessions[i]->run().end;
    callbacks[i] = [frame](std::uint32_t rows) {
      return !frame->cancelled && frame->exchange.Next(Phase::kPrefill, rows);
    };
  }
  const auto count = selected.size();
  std::array<Llm::GenerationSession*, Llm::kMaxBranches> generations{};
  std::array<ChatWork*, Llm::kMaxBranches> decoded{};
  std::size_t gcount = 0;
  for (auto* frame : selected_decode_) {
    if (!frame->exchange.Next(Phase::kDecode, 0)) {
      frame->Cancel();
      continue;
    }
    frame->native_touched = true;
    decoded[gcount] = frame;
    generations[gcount++] = frame->generation_session.get();
  }
  selected_decode_.clear();
  auto advanced = gcount ? cohort_model_->RunCausalWave(std::span(sessions).first(count),
                                                        std::span(callbacks).first(count),
                                                        std::span(generations).first(gcount), true)
                         : cohort_model_->RunPromptWave(std::span(sessions).first(count),
                                                        std::span(callbacks).first(count), true);
  if (!advanced) {
    if (server_.node().TakeHangCancelled()) return HangCohort("a prompt wave");
    Fail("the native chat prompt wave failed: " + advanced.error());
    return advanced;
  }
  if (server_.node().TakeHangCancelled()) return HangCohort("a prompt wave");
  // No session is prepared now: native completion and all peer history
  // publication precede cancellation, retirement or BeginChatGeneration.
  bool ages = false;
  std::array<bool, Llm::kMaxBranches> processed{};
  for (std::size_t i = 0; i < count; ++i) {
    if (sessions[i]->run().end > before[i]) {
      selected[i]->passed = 0;
      processed[selected[i]->slot] = true;
      ages = true;
    }
  }
  for (std::size_t i = 0; i < gcount; ++i) {
    if (!generations[i]->refused()) {
      decoded[i]->waited = 0;
      processed[decoded[i]->slot] = true;
    }
  }
  if (ages) {
    for (ChatWork* other : cohort_) {
      if (!other || processed[other->slot]) continue;
      if (other->stage == ChatWork::Stage::kGeneration)
        ++other->waited;
      else if (other->stage == ChatWork::Stage::kPrompt)
        ++other->passed;
    }
  }
  std::array<ChatWork*, Llm::kMaxBranches> refused{};
  std::size_t refusals = 0;
  for (std::size_t i = 0; i < count; ++i) {
    ChatWork& frame = *selected[i];
    const auto& result = sessions[i]->last_unit_result();
    if (!result) {
      const auto error = "the prompt could not be processed: " + result.error();
      if (sessions[i]->refused()) {
        frame.refusal = error;
        refused[refusals++] = &frame;
      } else {
        frame.error = Failure(500, error);
      }
    } else if (sessions[i]->done()) {
      if (auto begun = BeginChatGeneration(frame); !begun) return begun;
    }
  }
  for (std::size_t i = 0; i < gcount; ++i) {
    if (generations[i]->refused()) {
      decoded[i]->refusal = "the generation failed: " + generations[i]->refusal();
      refused[refusals++] = decoded[i];
    }
  }
  if (refusals != 0) WaitForCapacity(std::span(refused).first(refusals));
  Rebalance();
  return {};
}

std::uint64_t NodeBackend::SwapBytes(Llm& model) const {
  if (server_.resident() == &model) {
    return 0;
  }
  const auto used = [](Llm& llm) {
    std::uint64_t bytes = 0;
    // Branch wrappers and metadata are stable. Actual expiry, checkpoint
    // restoration and growth remain in their declared native prompt units.
    for (std::size_t i = 0; i < llm.branches(); ++i) {
      auto branch = llm.branch(i);
      if (branch) {
        bytes += (*branch)->state_snapshot_bytes();
      }
    }
    return bytes;
  };
  auto bytes = model.weight_read_bytes() + used(model);
  if (auto* outgoing = server_.resident(); outgoing != nullptr && outgoing->llm()) {
    bytes += used(static_cast<Llm&>(*outgoing));
  }
  return bytes;
}

Status NodeBackend::CheckCohort(std::span<api::CooperativeBackend::Work* const> work) const {
  if (cohort_model_ == nullptr || work.empty() || work.size() > cohort_.size()) {
    return std::unexpected("the chat cohort has no active model or exceeds its bound");
  }
  std::array<bool, Llm::kMaxBranches> seen{};
  for (auto* base : work) {
    if (base == nullptr) {
      return std::unexpected("the chat cohort has a null owner");
    }
    // Only Start on this backend constructs the caller-owned descriptors.
    const auto& frame = static_cast<const ChatWork&>(*base);
    if (&frame.owner != this || &frame.model != cohort_model_ || frame.slot >= cohort_.size() ||
        cohort_[frame.slot] != &frame || frame.retired || frame.terminal() || seen[frame.slot]) {
      return std::unexpected("the chat cohort contains an inactive, foreign or duplicate owner");
    }
    seen[frame.slot] = true;
  }
  if (std::ranges::count(seen, true) !=
      std::ranges::count_if(cohort_, [](const ChatWork* frame) { return frame != nullptr; })) {
    return std::unexpected("the chat unit omitted a claimed owner");
  }
  return {};
}

Status NodeBackend::SelectCohort() {
  std::array<Llm::Branch*, Llm::kMaxBranches> branches{};
  std::size_t count = 0;
  for (ChatWork* frame : cohort_) {
    if (frame != nullptr) {
      branches[count++] = &frame->branch;
    }
  }
  return server_.SelectRequestBranches(*cohort_model_, std::span(branches).first(count));
}

std::expected<std::unique_ptr<Llm::PromptSession>, std::string> NodeBackend::WorkPrompt(
    ChatWork& frame) {
  const auto budgeted = [&](std::expected<std::unique_ptr<Llm::PromptSession>, std::string> begun) {
    if (begun) {
      // Fresh generation samples one token from prompt logits; resumed
      // generation already owns its anchor and samples nothing here.
      (*begun)->SetContinuationRows(
          PromptTargetRows(frame.options.max_tokens, frame.generation.tokens.size(), frame.resume),
          frame.options.sampling && frame.options.sampling->temperature > 0);
    }
    return begun;
  };
  if (frame.literal != nullptr && frame.literal->score_prompt() && !frame.resume) {
    return budgeted(frame.branch.BeginScoringPrompt(
        frame.rendered.tokens(),
        [&frame](std::int32_t id, std::span<const float> row) {
          return frame.literal->PromptRow(frame.prompt_session->run().end, id, row) &&
                 frame.exchange.Continue();
        },
        frame.literal->prompt_started()));
  }
  const bool started = frame.literal != nullptr && frame.literal->prompt_started();
  return budgeted(frame.branch.BeginPrompt(
      frame.resume ? frame.resume_tokens : frame.rendered.tokens(), frame.rendered.stable_boundary,
      frame.literal != nullptr && !started, frame.resume || started));
}

Status NodeBackend::BeginChatGeneration(ChatWork& frame) {
  if (!frame.resume && !frame.continued) {
    frame.reused = frame.prompt_session->reused();
  }
  const bool stopped = frame.prompt_session->run().stopped;
  if (auto finished = frame.prompt_session->Finish(); !finished) {
    frame.error = Failure(500, "the prompt failed: " + finished.error());
    return {};
  }
  if (frame.literal != nullptr) {
    frame.literal->EndPrompt();
    if (frame.literal->error()) {
      frame.error = *frame.literal->error();
      return {};
    }
  }
  if (frame.rendered.max_tokens == 0) {
    frame.stage = ChatWork::Stage::kDone;
    return {};
  }
  if (stopped || frame.cancelled || !frame.exchange.Next(Phase::kDecode, 0)) {
    frame.stage = ChatWork::Stage::kDone;
    return {};
  }
  // After a preemption the rebuilt state continues the generation from its
  // anchor; nothing is chosen or streamed again.
  auto generation = frame.resume ? frame.branch.ResumeGeneration(frame.prompt_session->last(),
                                                                 frame.options, frame.generation)
                                 : frame.branch.BeginGeneration(frame.prompt_session->last(),
                                                                frame.options, frame.generation);
  if (!generation) {
    frame.error = Failure(500, "the generation could not start: " + generation.error());
    return {};
  }
  frame.resume = false;
  frame.switch_restore = false;
  frame.resume_tokens = {};
  frame.generation_session = std::move(*generation);
  frame.stage = ChatWork::Stage::kGeneration;
  frame.waited = 0;
  // Last logits are no longer borrowed by generation: Begin already chose
  // the first token. The callback and options remain in this stable frame.
  frame.prompt_session.reset();
  return {};
}

Cohort NodeBackend::Snapshot() const {
  Cohort cohort{};
  for (std::size_t i = 0; i < cohort_.size(); ++i) {
    if (const ChatWork* frame = cohort_[i]; frame != nullptr) {
      cohort[i] = {.present = true,
                   .terminal = frame->terminal(),
                   .admitted = frame->admitted,
                   .wait = frame->wait};
    }
  }
  return cohort;
}

void NodeBackend::Apply(const Cohort& cohort) {
  for (std::size_t i = 0; i < cohort_.size(); ++i) {
    if (cohort_[i] != nullptr) {
      cohort_[i]->wait = cohort[i].wait;
    }
  }
}

void NodeBackend::WaitForCapacity(std::span<ChatWork* const> refused) {
  std::vector<std::size_t> slots;
  for (const ChatWork* frame : refused) {
    slots.push_back(frame->slot);
  }
  Cohort cohort = Snapshot();
  // The node's reclaim order first (Server::Reclaim): plans and graphs,
  // idle conversations spilled, in the order's priority, for what each
  // refused member asked: all of it or nothing, member by member, so one
  // member's need that cannot be met does not keep another's from it
  // (those relieved run again; the rest are refused again and wait).
  bool reclaimed = false;
  for (const ChatWork* frame : refused) {
    const std::uint64_t needed =
        std::max<std::uint64_t>(frame->model.RefusedBytes(frame->branch), kExtentBytes);
    reclaimed = server_.Reclaim(needed, true, "conversation state refused") >= needed || reclaimed;
  }
  CapacityDecision decision = OnCapacityRefused(cohort, slots, reclaimed);
  Apply(cohort);
  if (decision.reclaim) {
    return;  // the refused members run again
  }
  for (const std::size_t slot : decision.refuse) {
    ChatWork& frame = *cohort_[slot];
    frame.error = Failure(500, frame.refusal);
  }
  // Positions and slots only (D-014).
  for (const ChatWork* frame : refused) {
    // The tokens its state holds: a generation publishes its history only
    // when it ends, and its anchor is not processed yet.
    const std::size_t held =
        frame->generation_session != nullptr
            ? frame->rendered.tokens().size() + frame->generation.tokens.size() - 1
            : frame->branch.history().size();
    if (frame->wait == CapacityWait::kBlocked) {
      Say(log_, std::format("{}'s request in slot {} waits for conversation-state capacity at "
                            "{} tokens",
                            frame->model.name(), frame->slot, held));
    } else if (frame->error) {
      Say(log_, std::format("{}'s request in slot {} does not fit the state's capacity alone at "
                            "{} tokens: refused",
                            frame->model.name(), frame->slot, held));
    }
  }
  if (decision.preempt) {
    Preempt(*cohort_[*decision.preempt]);
  }
}

void NodeBackend::Preempt(ChatWork& frame) {
  Status finished;
  if (frame.prompt_session != nullptr) {
    frame.prompt_session->Cancel();
    finished = frame.prompt_session->Finish();
    frame.prompt_session.reset();
  } else if (frame.generation_session != nullptr) {
    frame.generation_session->Cancel();
    finished = frame.generation_session->Finish();
    frame.generation_session.reset();
    if (finished) {
      // The prompt and the generated tokens the state holds; the anchor
      // stays in frame.generation, already streamed.
      frame.resume_tokens = frame.branch.history();
      (void)frame.FundTokens(true);
      frame.resume = true;
      frame.stage = ChatWork::Stage::kPrompt;
    }
  }
  if (!finished) {
    frame.error = Failure(500, "the request could not release its state: " + finished.error());
    return;
  }
  const std::size_t held = frame.branch.history().size();
  // Spill, not clear: its state written to its slot's spill file and
  // restored exactly when it begins again, so it continues as if never
  // set aside (Llm::SpillSetAside). Only a state that cannot be spilled
  // is cleared, and then rebuilt by prefill.
  if (auto spilled = frame.model.SpillSetAside(frame.branch); spilled) {
    Say(log_, std::format("{}'s request in slot {} set aside for its peers: its state spilled "
                          "at {} tokens, restored when they finish",
                          frame.model.name(), frame.slot, held));
    return;
  }
  if (auto cleared = frame.branch.Clear(); !cleared) {
    frame.error = Failure(500, "the request could not release its state: " + cleared.error());
    return;
  }
  Say(log_, std::format("{}'s request in slot {} released its state at {} tokens for its "
                        "peers; it is rebuilt when they finish",
                        frame.model.name(), frame.slot, held));
}

void NodeBackend::Rebalance() {
  Cohort cohort = Snapshot();
  const auto restart = NextRestart(cohort);
  Apply(cohort);
  if (!restart) {
    return;
  }
  ChatWork& frame = *cohort_[*restart];
  // A generation set aside resumes from its spilled state: its prompt
  // only restores it (BeginPrompt's `resume`).
  auto prompt = WorkPrompt(frame);
  if (!prompt) {
    frame.error = Failure(500, "the request could not begin again: " + prompt.error());
    return;
  }
  frame.prompt_session = std::move(*prompt);
  frame.stage = ChatWork::Stage::kPrompt;
}

std::expected<void, std::string> NodeBackend::HangCohort(std::string_view what) {
  hang_reset_ = cohort_model_;
  std::size_t members = 0;
  for (ChatWork* frame : cohort_) {
    if (frame != nullptr && !frame->retired) {
      frame->error = HungFailure();
      ++members;
    }
  }
  selected_prompt_ = nullptr;
  selected_prefill_count_ = 0;
  selected_decode_.clear();
  Say(log_, std::format("hang recovery, rung 2 (reset the model in place): {} failed after its "
                        "hung work was cancelled; its {} requests fail, and {} is reset once "
                        "they retire and the cancellation drained",
                        what, members, cohort_model_ != nullptr ? cohort_model_->name() : ""));
  return {};
}

}  // namespace llmp::runtime::service_detail
