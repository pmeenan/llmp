// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The concrete LLM's host conversation ownership: branches, prompt and
// generation sessions, turn checkpoints, spills and kept conversations.
// Native work and state are implemented by the same Llm in llm_execution.cc,
// through the shared runner and slot lifecycle, in every build profile.

#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
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
#include <type_traits>
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

using serving_support::Error;
using serving_support::KeptId;
using serving_support::KeptRanges;
using serving_support::kExtent;
using serving_support::LiveRanges;
using serving_support::Seconds;
using serving_support::SteadyAt;
using serving_support::UnixMs;

namespace ja = llmp::artifact;

// strftime_now's clock for an interpreted chat template: local time, as
// Python's datetime.now() gives it.
std::optional<chat::jinja::CivilTime> LocalTime() {
  const std::time_t now = std::time(nullptr);
  std::tm tm{};
  if (localtime_r(&now, &tm) == nullptr) {
    return std::nullopt;
  }
  return chat::jinja::CivilTime{.year = tm.tm_year + 1900,
                                .month = tm.tm_mon + 1,
                                .day = tm.tm_mday,
                                .hour = tm.tm_hour,
                                .minute = tm.tm_min,
                                .second = tm.tm_sec,
                                .weekday = tm.tm_wday,
                                .yearday = tm.tm_yday};
}

using llm_detail::HistoryFundingGuard;

}  // namespace

// ---------------------------------------------------------------- an LLM

void Llm::SetTokenMemory(RequestMemory& memory) {
  base::Check(token_history_bytes() == 0, "changing a model's token pool with live history");
  for (std::size_t slot = 0; slot < branches(); ++slot) {
    const auto b = branch(slot);
    base::Check(b && !(*b)->generation_active_ && (*b)->prompt_session_ == nullptr &&
                    !(*b)->funding_history_,
                "changing a model's token pool with an active session");
  }
  token_memory_ = &memory;
}

std::uint64_t Llm::token_history_bytes() const {
  std::uint64_t bytes = 0;
  for (std::uint32_t slot = 0; slot < branch_count_; ++slot) {
    const Branch& b = slot == 0 ? default_branch_ : *extra_branches_[slot - 1];
    bytes += b.history_charge_.bytes() + b.saved_history_charge_.bytes();
  }
  return bytes;
}

std::uint64_t Llm::IdleHistoryBytes(const Branch& branch) const {
  CheckBranch(branch);
  return branch.history_charge_.bytes();
}

bool Llm::HistoryReclaimable(const Branch& branch) const {
  return BranchIdle(branch) && !branch.HeldByContinuation() && !branch.funding_history_ &&
         !branch.saved_valid_;
}

bool Llm::ReserveTokens(Branch& branch, std::vector<std::int32_t>& tokens, MemoryCharge& charge,
                        std::size_t capacity) {
  if (capacity <= tokens.capacity()) {
    return true;
  }
  const bool was_funding = std::exchange(branch.funding_history_, true);
  // Preserve amortized growth under pressure too: exact one-token growth
  // would repeatedly copy the whole conversation near the budget limit.
  const std::size_t grown =
      std::max(capacity, std::min(std::size_t{context_} + 1, tokens.capacity() * 2));
  const bool funded = ReserveTokenStorage(tokens, charge, *token_memory_, grown);
  token_memory_->Settle(0, std::chrono::milliseconds{0});
  branch.funding_history_ = was_funding;
  return funded;
}

void Llm::ReleaseTokens(std::vector<std::int32_t>& tokens, MemoryCharge& charge) {
  ReleaseTokenStorage(tokens, charge);
  token_memory_->Settle(0, std::chrono::milliseconds{0});
}

void Llm::PublishTokens(Branch& branch, std::vector<std::int32_t>& tokens, MemoryCharge& charge,
                        std::size_t size) {
  base::Check(size <= tokens.size(), "publishing beyond the funded history");
  if (size == 0) {
    ReleaseTokens(tokens, charge);
    ReleaseTokens(branch.history_, branch.history_charge_);
    return;
  }
  tokens.resize(size);
  branch.history_.swap(tokens);
  std::swap(branch.history_charge_, charge);
  ReleaseTokens(tokens, charge);
}

void Llm::Branch::InvalidateStateSnapshot() {
  snapshot_issuer_ = std::make_shared<const std::byte>();
  saved_valid_ = false;
  saved_score_rows_ = {};
  model_.ReleaseTokens(saved_history_, saved_history_charge_);
  saved_ranges_.clear();
}

Llm::Branch::~Branch() {
  snapshot_issuer_.reset();  // invalidate capabilities before releasing branch metadata
  base::Check(!generation_active_, "a branch with an active generation was destroyed");
  base::Check(prompt_session_ == nullptr, "a branch with an active prompt was destroyed");
}

Status Llm::Branch::Clear() { return model_.Clear(*this); }

Status Llm::Branch::ReleaseIdleState() { return model_.ReleaseIdleState(*this); }

bool Llm::Branch::spilled() const { return model_.SpilledFor(*this); }

void Llm::Branch::Forget() { model_.Forget(*this); }

Status Llm::Branch::Prefill(std::span<const std::int32_t> tokens, std::vector<float>& last,
                            const PrefillGoOn& go_on, PrefillRun* run) {
  return model_.Prefill(*this, tokens, last, go_on, run);
}

Status Llm::Branch::ScorePrompt(
    std::span<const std::int32_t> tokens, std::vector<float>& last,
    const std::function<bool(std::int32_t, std::span<const float>)>& on_row,
    const PrefillGoOn& go_on, PrefillRun* run) {
  return model_.ScorePrompt(*this, tokens, last, on_row, go_on, run);
}

Status Llm::Branch::PreparePrompt(std::span<const std::int32_t> tokens,
                                  std::uint32_t stable_boundary, std::vector<float>& last,
                                  std::uint32_t& reused, const PrefillGoOn& go_on, PrefillRun* run,
                                  bool fresh, bool resume) {
  return model_.PreparePrompt(*this, tokens, stable_boundary, last, reused, go_on, run, fresh,
                              resume);
}

std::expected<std::unique_ptr<Llm::PromptSession>, std::string> Llm::Branch::BeginPrompt(
    std::span<const std::int32_t> tokens, std::uint32_t stable_boundary, bool fresh,
    bool resume) & {
  return model_.BeginPrompt(*this, tokens, stable_boundary, fresh, resume);
}

std::expected<std::unique_ptr<Llm::PromptSession>, std::string> Llm::Branch::BeginScoringPrompt(
    std::span<const std::int32_t> tokens,
    std::function<bool(std::int32_t, std::span<const float>)> on_row, bool resume) & {
  return model_.BeginPrompt(*this, tokens, 0, !resume, resume, true, std::move(on_row));
}

std::expected<std::unique_ptr<Llm::GenerationSession>, std::string> Llm::Branch::BeginGeneration(
    const std::vector<float>& last, const GenerateOptions& options, Generation& out) & {
  return model_.BeginGeneration(*this, last, options, out);
}

std::expected<std::unique_ptr<Llm::GenerationSession>, std::string> Llm::Branch::ResumeGeneration(
    const std::vector<float>& last, const GenerateOptions& options, Generation& out) & {
  return model_.BeginGeneration(*this, last, options, out, true);
}

Status Llm::Branch::Generate(const std::vector<float>& last, const GenerateOptions& options,
                             Generation& out) {
  return model_.Generate(*this, last, options, out);
}

std::uint64_t Llm::Branch::state_snapshot_bytes() const {
  model_.CheckBranch(*this);
  return model_.UsedStateBytesFor(*this);
}

Status Llm::Branch::SaveState(void* host) { return model_.SaveState(*this, host); }

Status Llm::Branch::RestoreState(void* host) { return model_.RestoreState(*this, host); }

Status Llm::Clear() { return default_branch_.Clear(); }

void Llm::Forget() { default_branch_.Forget(); }

Status Llm::Prefill(std::span<const std::int32_t> tokens, std::vector<float>& last,
                    const PrefillGoOn& go_on, PrefillRun* run) {
  return default_branch_.Prefill(tokens, last, go_on, run);
}

Status Llm::ScorePrompt(std::span<const std::int32_t> tokens, std::vector<float>& last,
                        const std::function<bool(std::int32_t, std::span<const float>)>& on_row,
                        const PrefillGoOn& go_on, PrefillRun* run) {
  return default_branch_.ScorePrompt(tokens, last, on_row, go_on, run);
}

Status Llm::PreparePrompt(std::span<const std::int32_t> tokens, std::uint32_t stable_boundary,
                          std::vector<float>& last, std::uint32_t& reused, const PrefillGoOn& go_on,
                          PrefillRun* run, bool fresh, bool resume) {
  return default_branch_.PreparePrompt(tokens, stable_boundary, last, reused, go_on, run, fresh,
                                       resume);
}

std::uint64_t Llm::turn_checkpoint_bytes() const { return default_branch_.turn_checkpoint_bytes(); }

std::expected<std::unique_ptr<Llm::GenerationSession>, std::string> Llm::BeginGeneration(
    const std::vector<float>& last, const GenerateOptions& options, Generation& out) & {
  return default_branch_.BeginGeneration(last, options, out);
}

Status Llm::Generate(const std::vector<float>& last, const GenerateOptions& options,
                     Generation& out, bool resume) {
  return Generate(default_branch_, last, options, out, resume);
}

std::uint64_t Llm::state_snapshot_bytes() const { return default_branch_.state_snapshot_bytes(); }

Status Llm::SaveState(void* host) { return default_branch_.SaveState(host); }

Status Llm::RestoreState(void* host) { return default_branch_.RestoreState(host); }

std::expected<EncodedText, std::string> Llm::EncodeText(std::string_view text,
                                                        RequestMemory* memory,
                                                        std::uint64_t* needed) const {
  MemoryCharge work_charge;
  EncodedText encoded;
  auto& ids = encoded.tokens;
  const auto output = tokenizer_->OutputTokensBound(text.size(), true);
  const tokenizer::EncodeOptions options{.special = tokenizer::SpecialTokens::kUserDefinedOnly,
                                         .add_bos_eos = true,
                                         .max_bytes = text.size(),
                                         .max_tokens = static_cast<std::size_t>(output)};
  std::uint64_t capacity = 0, output_bytes = 0, need = 0;
  const auto working = tokenizer_->WorkingBytes(text);
  if (__builtin_add_overflow(output, std::uint64_t{1}, &capacity) || capacity > ids.max_size() ||
      __builtin_mul_overflow(capacity, std::uint64_t{sizeof(tokenizer::TokenId)}, &output_bytes) ||
      __builtin_add_overflow(working, output_bytes, &need)) {
    if (needed != nullptr) *needed = std::numeric_limits<std::uint64_t>::max();
    return Error("encoding output capacity overflows its representation");
  }
  if (memory != nullptr) {
    if (!work_charge.Add(*memory, working) || !encoded.charge.Add(*memory, output_bytes)) {
      if (needed != nullptr) {
        *needed = need;
      }
      return Error(std::format("encoding the text needs {} bytes of request memory", need));
    }
  }
  // Exact paid capacity includes the endpoint's extra BOS, so neither source
  // insertion nor the final prepend can trigger geometric vector growth.
  if (auto reserved = tokenizer_->ReserveOutput(text, options, ids, 1); !reserved)
    return Error(reserved.error().ToString());
  if (auto r = tokenizer_->Encode(text, options, ids); !r) {
    return Error(r.error().ToString());
  }
  if (tokenizer_->bos() && (ids.empty() || ids.front() != *tokenizer_->bos())) {
    ids.insert(ids.begin(), *tokenizer_->bos());  // BOS first, as the references fed it
  }
  return encoded;
}

RenderCharge Llm::RenderChargeFor(std::uint64_t message_bytes, std::size_t messages) const {
  constexpr std::uint64_t kMiB = std::uint64_t{1} << 20U;
  // Four times the text, and a message's framing (role headers, turn
  // markers) for each message, however short.
  const std::uint64_t framing = std::uint64_t{messages} * kRenderBytesPerMessage;
  const std::uint64_t scaled = message_bytes > (std::numeric_limits<std::uint64_t>::max() / 8) ||
                                       framing > (std::numeric_limits<std::uint64_t>::max() / 8)
                                   ? std::numeric_limits<std::uint64_t>::max() / 4
                                   : (4 * message_bytes) + framing;
  RenderCharge c;
  c.output = std::min<std::uint64_t>(render_bytes_, scaled + kMinRenderBytes);
  c.bytes = c.output;
  if (template_ && template_->interprets()) {
    c.live = std::min<std::uint64_t>(template_->max_live_bytes(), scaled + (16 * kMiB));
    c.bytes += c.output + c.live + (2 * message_bytes);
  }
  return c;
}

std::expected<std::vector<std::int32_t>, std::string> Llm::RenderChat(
    const chat::Conversation& conversation, std::uint32_t* stable_boundary,
    const ChatRenderOptions& options, ChatRenderFailure* failure) const {
  if (stable_boundary != nullptr) {
    *stable_boundary = 0;
  }
  if (failure != nullptr) {
    *failure = ChatRenderFailure::kOther;
  }
  if (!template_) {
    return Error(std::format("{} has no chat template (D-067)", name_));
  }
  if (auto checked = CheckConversation(conversation); !checked) return Error(checked.error());
  auto rendered = template_->Render(conversation, LocalTime(), options.cancelled);
  if (!rendered) {
    if (failure != nullptr && rendered.error().cancelled) {
      *failure = ChatRenderFailure::kCancelled;
    } else if (failure != nullptr && rendered.error().bound &&
               conversation.max_render_bytes >= render_bytes_) {
      // Longer than any prompt that fits the context could render.
      *failure = ChatRenderFailure::kTooLong;
    }
    return Error(rendered.error().ToString());
  }
  if (options.reasoning != nullptr || options.may_reason != nullptr) {
    bool reasoning = false, may_reason = false;
    const auto* format = template_->tool_calls();
    if (format != nullptr && !format->reasoning_begin.empty()) {
      for (const auto& boundary : rendered->boundaries) {
        if (boundary.kind != chat::BoundaryKind::kGenerationPrompt) continue;
        const auto suffix = std::string_view(rendered->text).substr(boundary.offset);
        const auto begin = suffix.rfind(format->reasoning_begin);
        const auto end = suffix.rfind(format->reasoning_end);
        reasoning =
            begin != std::string_view::npos && (end == std::string_view::npos || begin > end);
        // Gemma 4's thinking turn renders no channel: the model opens it.
        may_reason = begin == std::string_view::npos && end == std::string_view::npos &&
                     !format->reasoning_end.empty();
      }
    }
    if (options.reasoning != nullptr) *options.reasoning = reasoning;
    if (options.may_reason != nullptr) *options.may_reason = may_reason;
  }
  // The whole-fragment working set and bounded output are charged together.
  MemoryCharge charge;
  if (options.memory != nullptr) {
    const auto output = std::min<std::uint64_t>(
        tokenizer_->OutputTokensBound(rendered->text.size()), options.max_tokens);
    std::uint64_t need = 0, output_bytes = 0, span_bytes = 0;
    if (__builtin_mul_overflow(output, std::uint64_t{2 * sizeof(tokenizer::TokenId)},
                               &output_bytes) ||
        __builtin_mul_overflow(static_cast<std::uint64_t>(rendered->specials.size()),
                               std::uint64_t{2 * sizeof(std::size_t)}, &span_bytes) ||
        __builtin_add_overflow(tokenizer_->WorkingBytes(rendered->text), output_bytes, &need) ||
        __builtin_add_overflow(need, span_bytes, &need))
      need = std::numeric_limits<std::uint64_t>::max();
    if (!charge.Add(*options.memory, need)) {
      if (failure != nullptr) {
        *failure = ChatRenderFailure::kMemory;
      }
      if (options.memory_needed != nullptr) {
        *options.memory_needed = need;
      }
      return Error(std::format("tokenizing the rendering needs {} bytes of request memory", need));
    }
  }
  // The rendering's own bytes (already within its bound) and the caller's
  // token bound, not the tokenizer's library defaults: a prompt is bounded
  // by the model's context (D-102).
  std::vector<tokenizer::TokenId> ids;
  std::vector<std::size_t> span_tokens;
  if (auto r = tokenizer_->EncodeMarked(rendered->text, rendered->specials,
                                        {.special = tokenizer::SpecialTokens::kUserDefinedOnly,
                                         .add_bos_eos = false,
                                         .max_bytes = rendered->text.size(),
                                         .max_tokens = options.max_tokens},
                                        ids, stable_boundary == nullptr ? nullptr : &span_tokens);
      !r) {
    if (failure != nullptr && r.error().rule == tokenizer::Rule::kOutputTooLarge) {
      *failure = ChatRenderFailure::kTooLong;
    }
    if (failure != nullptr && r.error().rule == tokenizer::Rule::kCancelled) {
      *failure = ChatRenderFailure::kCancelled;
    }
    return Error(r.error().ToString());
  }
  if (stable_boundary != nullptr) {
    for (const chat::Boundary& boundary : rendered->boundaries) {
      if (boundary.kind != chat::BoundaryKind::kGenerationPrompt) {
        continue;
      }
      for (std::size_t i = 0; i < rendered->specials.size(); ++i) {
        if (rendered->specials[i].offset == boundary.offset && span_tokens[i] < ids.size()) {
          *stable_boundary = static_cast<std::uint32_t>(span_tokens[i]);
          break;
        }
      }
    }
  }
  return ids;
}

std::string Llm::Detokenize(std::span<const std::int32_t> tokens) const {
  std::string text;
  if (auto r = tokenizer_->Decode(tokens, {}, text); !r) {
    return "(not decodable)";
  }
  return text;
}

void Llm::ReserveBranchSamplingScratch(Branch& branch, std::size_t capacity) {
  CheckBranch(branch);
  branch.scratch_.reserve(capacity);
}
void Llm::DropBranchSamplingScratch(Branch& branch) {
  CheckBranch(branch);
  std::vector<execution::SamplingCandidate>().swap(branch.scratch_);
}

std::expected<std::span<const float>, std::string> Llm::ConstrainedRow(
    Branch& branch, std::span<const float> row, chat::ToolSampling* constraint) {
  auto* mask = constraint ? constraint : branch.tool_sampling_;
  if (!mask) return row;
  // A mask must not erase invalid native output or reclassify it as a
  // request-local candidate refusal. Choose and Keep share this boundary.
  if (auto valid = execution::ValidateLogits(row); !valid)
    return Error(std::string(execution::SamplingErrorName(valid.error())));
  const auto filtered = mask->Mask(row, branch.constrained_logits_);
  const auto finite = [](float value) { return std::isfinite(value); };
  if (branch.tool_sampling_ && std::ranges::none_of(filtered, finite)) {
    // Verification uses a tentative lexical cursor; the request itself owns
    // this refusal. Invalid native rows never acquire this classification.
    branch.tool_sampling_->Refuse();
  }
  return filtered;
}

std::expected<std::int32_t, std::string> Llm::Choose(Branch& branch, std::span<const float> row,
                                                     std::uint64_t position,
                                                     chat::ToolSampling* constraint) {
  CheckBranch(branch);
  auto constrained = ConstrainedRow(branch, row, constraint);
  if (!constrained) return std::unexpected(constrained.error());
  row = *constrained;
  if (!branch.sampling_) {
    auto chosen = execution::Greedy(row);
    if (!chosen) return Error(std::string(execution::SamplingErrorName(chosen.error())));
    return *chosen;
  }
  if (auto prepared = PrepareSamplingScratchFor(branch, row.size()); !prepared)
    return std::unexpected(prepared.error());
  auto token =
      execution::Sample(row, *branch.sampling_,
                        {.seed = branch.seed_, .stream = 0, .position = position}, branch.scratch_);
  if (!token) {
    return Error(std::format("sampling: {}", execution::SamplingErrorName(token.error())));
  }
  return *token;
}

std::expected<bool, std::string> Llm::Keep(Branch& branch, std::span<const float> row,
                                           std::int32_t draft, std::uint64_t position,
                                           std::int32_t& next, chat::ToolSampling* constraint) {
  CheckBranch(branch);
  auto constrained = ConstrainedRow(branch, row, constraint);
  if (!constrained) return std::unexpected(constrained.error());
  row = *constrained;
  if (!branch.sampling_) {
    auto chosen = execution::Greedy(row);
    if (!chosen) return Error(std::string(execution::SamplingErrorName(chosen.error())));
    next = *chosen;
    return next == draft;
  }
  auto verdict = execution::VerifyDraft(row, draft, *branch.sampling_,
                                        {.seed = branch.seed_, .stream = 0, .position = position},
                                        branch.scratch_);
  if (!verdict) {
    return Error(std::format("sampling: {}", execution::SamplingErrorName(verdict.error())));
  }
  next = verdict->token;
  return verdict->accepted;
}

namespace {
struct ChatTemplateCandidate {
  std::optional<chat::ChatTemplate> chosen;
  std::vector<std::int32_t> stops;
  std::size_t longest = 1, render_bytes = 0;
};

std::expected<ChatTemplateCandidate, std::string> MakeChatTemplate(
    const tokenizer::Tokenizer& tokenizer, std::uint32_t context, std::string_view text) {
  ChatTemplateCandidate out;
  out.longest = tokenizer.longest_token_bytes();
  out.render_bytes = static_cast<std::size_t>(RenderBytes(
      context, out.longest, tokenizer.normalization() == tokenizer::Normalization::kNfc));
  chat::jinja::Limits limits;
  limits.max_output_bytes = out.render_bytes;
  limits.max_string_bytes = std::max(limits.max_string_bytes, out.render_bytes);
  limits.max_live_bytes = std::max(limits.max_live_bytes, 2 * out.render_bytes);
  auto chosen = chat::ChatTemplate::ForText(text, chat::TokenFacts::From(tokenizer), limits);
  if (!chosen) return Error(chosen.error());
  out.chosen = std::move(*chosen);
  auto stops = chat::StopTokens(out.chosen->stop(), tokenizer);
  if (!stops) return Error(stops.error().ToString());
  out.stops.assign(stops->begin(), stops->end());
  if (tokenizer.eos() && std::ranges::find(out.stops, *tokenizer.eos()) == out.stops.end())
    out.stops.push_back(*tokenizer.eos());
  return out;
}

void ReportChatTemplate(std::string_view name, const chat::ChatTemplate& chosen) {
  if (chosen.how() != chat::ChatTemplate::How::kNativeByHash)
    std::println(stderr, "{}: chat template {} rendered {} (D-067)", name,
                 chosen.sha256().substr(0, 12),
                 chosen.how() == chat::ChatTemplate::How::kNativeByProbe
                     ? std::format("natively by {}, equal on its probe corpus", chosen.name())
                     : std::string("by the sandboxed template interpreter"));
}
}  // namespace

Status Llm::UseTemplate(std::string_view text) {
  auto candidate = MakeChatTemplate(*tokenizer_, context_, text);
  if (!candidate) return Error(candidate.error());
  template_.swap(candidate->chosen);
  stops_.swap(candidate->stops);
  longest_token_ = candidate->longest;
  render_bytes_ = candidate->render_bytes;
  ReportChatTemplate(name_, *template_);
  return {};
}

std::uint64_t Llm::startup_host_bytes() const {
  return startup_host_bytes_ +
         (execution_metadata_ ? execution_metadata_->capacity() +
                                    engine::ExecutionMetadataAllowance::OverheadBytes()
                              : 0) +
         (chat_charge_ ? chat_charge_->bytes + engine::ExecutionMetadataAllowance::OverheadBytes()
                       : 0);
}

Status Llm::UseChatAssets(const ja::Artifact& artifact, const config::ModelEntry& entry,
                          std::uint64_t startup_cap) {
  // Reader funding precedes every payload read. Server lends its existing live
  // serialized parser payment; a direct caller or expired view pays locally.
  // Diagnostic cap==0 retains the explicit caller-funded construction mode.
  std::unique_ptr<StartupArtifactPayment> reader;
  if (startup_cap != 0) {
    auto bytes = ChatReaderBytes();
    if (!bytes) return Error(bytes.error());
    if (!chat_reader_.Covers(node_->catalog(), node_->domain(), *bytes)) {
      auto paid = StartupArtifactPayment::AdmitReader(node_->catalog(), node_->domain(), *bytes,
                                                      startup_cap,
                                                      platform::AvailableMemoryBytes().value_or(0));
      if (!paid) return Error(paid.error());
      reader = std::move(*paid);
    }
  }
  auto assets = ReadChatAssets(artifact, entry);
  if (!assets) return Error(assets.error());
  struct ChatCandidate {
    // Declared first: every candidate/old payload dies before its live charge.
    std::unique_ptr<serving_support::Paid> charge;
    std::unique_ptr<tokenizer::Tokenizer> tokenizer;
    ReasoningSettings reasoning;
    std::optional<std::int32_t> think_start, think_end;
    ChatTemplateCandidate chat;
  } candidate;
  if (startup_cap != 0) {
    auto bytes = tokenizer::Tokenizer::ConstructionBytes(assets->spec);
    if (!bytes) return Error(bytes.error().ToString());
    constexpr chat::jinja::Limits limits;
    const std::uint64_t template_size =
        assets->chat_template
            ? std::min<std::uint64_t>(assets->chat_template->size(), limits.max_template_bytes)
            : 0;
    const std::uint64_t nodes = std::min<std::uint64_t>(template_size + 1, limits.max_nodes);
    constexpr std::uint64_t kProbeScratchBytes = std::uint64_t{32} << 20U;
    // AST/source/probe plus candidate ownership and stop-token vector growth.
    const std::uint64_t template_bytes = 2048 * nodes + 4 * template_size + kProbeScratchBytes +
                                         sizeof(ChatCandidate) + sizeof(serving_support::Paid) +
                                         4096;
    if (__builtin_add_overflow(*bytes, template_bytes, &*bytes))
      return Error("chat asset construction payment overflows");
    // The two resolved settings own their text and override provenance. Include
    // formatting/copy growth before either setting is constructed.
    std::uint64_t reasoning = 1024;
    if (__builtin_add_overflow(reasoning, entry.name.size(), &reasoning))
      return Error("chat reasoning payment overflows");
    for (const auto key : {"reasoning_start", "reasoning_end"}) {
      const auto setting = entry.overrides.find(key);
      if (setting == entry.overrides.end()) continue;
      const auto* text = std::get_if<std::string>(&setting->second);
      if (text && __builtin_add_overflow(reasoning, text->size(), &reasoning))
        return Error("chat reasoning payment overflows");
    }
    if (__builtin_mul_overflow(reasoning, std::uint64_t{8}, &reasoning) ||
        __builtin_add_overflow(*bytes, reasoning, &*bytes))
      return Error("chat reasoning payment overflows");
    if (!chat_startup_) {
      std::uint64_t total = 0;
      if (__builtin_add_overflow(*bytes, engine::ExecutionMetadataAllowance::OverheadBytes(),
                                 &total))
        return Error("chat asset payment overhead overflows");
      auto pool = serving_support::AdmitExecutionMetadata(*node_, total, startup_cap);
      if (!pool) return Error(pool.error());
      chat_startup_ = std::move(*pool);
    }
    if (!serving_support::ChargeExecutionMetadata(chat_startup_, *bytes, node_, startup_cap))
      return Error("chat asset construction exceeds the node execution cap or available memory");
    candidate.charge = std::make_unique<serving_support::Paid>(chat_startup_, *bytes);
  }
  auto created = tokenizer::Tokenizer::Create(std::move(assets->spec));
  if (!created) return Error(std::format("{}: {}", assets->from, created.error().ToString()));
  candidate.tokenizer = std::make_unique<tokenizer::Tokenizer>(std::move(*created));
  auto reasoning = ResolveReasoning(
      entry, [&](std::string_view text) { return candidate.tokenizer->Find(text); });
  if (!reasoning) return Error(reasoning.error());
  candidate.reasoning = std::move(*reasoning);
  const auto find = [&](const std::string& text) -> std::optional<std::int32_t> {
    return text.empty() ? std::nullopt : candidate.tokenizer->Find(text);
  };
  candidate.think_start = find(candidate.reasoning.start.value);
  candidate.think_end = find(candidate.reasoning.end.value);
  if (!assets->chat_template)
    return Error(
        std::format("{} keeps no chat template and models.{}.chat_template is not set, "
                    "so it has no chat turns",
                    assets->from, entry.name));
  auto chosen = MakeChatTemplate(*candidate.tokenizer, context_, *assets->chat_template);
  if (!chosen) return Error(chosen.error());
  candidate.chat = std::move(*chosen);
  // Publication cannot fail or allocate. The candidate takes every old payload
  // and its independent charge, keeping old+new funded through the last swap.
  static_assert(std::is_nothrow_swappable_v<decltype(template_)>);
  static_assert(std::is_nothrow_swappable_v<Setting<std::string>>);
  using std::swap;
  chat_charge_.swap(candidate.charge);
  tokenizer_.swap(candidate.tokenizer);
  template_.swap(candidate.chat.chosen);
  stops_.swap(candidate.chat.stops);
  swap(longest_token_, candidate.chat.longest);
  swap(render_bytes_, candidate.chat.render_bytes);
  swap(think_start_, candidate.think_start);
  swap(think_end_, candidate.think_end);
  swap(settings_.reasoning_start, candidate.reasoning.start);
  swap(settings_.reasoning_end, candidate.reasoning.end);
  ReportChatTemplate(name_, *template_);
  return {};
}

Status Llm::PrepareBranches(std::uint32_t count, std::uint32_t draft_depth) {
  if (branches_prepared_ || count == 0 || count > kMaxBranches || draft_depth == 0 ||
      default_branch_.generation_active_ || !default_branch_.history_.empty()) {
    return Error("native branches must be prepared once before conversation use");
  }
  for (std::uint32_t slot = 1; slot < count; ++slot) {
    extra_branches_[slot - 1] = std::unique_ptr<Branch>(new Branch(*this, slot));
    extra_branches_[slot - 1]->decoding_ = execution::AdaptiveDepth(draft_depth, depth_cost_ratio_);
    extra_branches_[slot - 1]->saved_decoding_ =
        execution::AdaptiveDepth(draft_depth, depth_cost_ratio_);
  }
  default_branch_.decoding_ = execution::AdaptiveDepth(draft_depth, depth_cost_ratio_);
  default_branch_.saved_decoding_ = execution::AdaptiveDepth(draft_depth, depth_cost_ratio_);
  branch_count_ = count;
  branches_prepared_ = true;
  return {};
}

std::expected<Llm::Branch*, std::string> Llm::branch(std::size_t index) & {
  if (index >= branch_count_) {
    return Error("the model has no native conversation slot at this index");
  }
  return index == 0 ? &default_branch_ : extra_branches_[index - 1].get();
}

void Llm::CheckBranch(const Branch& branch) const {
  base::Check(&branch.model_ == this && branch.slot_ < branch_count_ &&
                  (branch.slot_ == 0 ? &branch == &default_branch_
                                     : &branch == extra_branches_[branch.slot_ - 1].get()),
              "conversation branch does not belong to this model's native slot");
}

std::uint32_t Llm::BranchIndex(const Branch& branch) const {
  CheckBranch(branch);
  return branch.slot_;
}

bool Llm::AnyBranchHasRetainedState() const {
  if (default_branch_.HasRetainedState()) {
    return true;
  }
  for (std::uint32_t slot = 1; slot < branch_count_; ++slot) {
    if (extra_branches_[slot - 1]->HasRetainedState()) {
      return true;
    }
  }
  return false;
}

bool Llm::BranchIdle(const Branch& branch) const {
  CheckBranch(branch);
  return !branch.generation_active_ && branch.prompt_session_ == nullptr &&
         !branch.funding_history_ && !LeasedFor(branch);
}

std::uint64_t Llm::StateBytes(const Branch& branch) const {
  CheckBranch(branch);
  return UsedStateBytesFor(branch);
}

std::uint64_t Llm::ResidentStateBytes(const Branch& branch) const {
  CheckBranch(branch);
  return SpilledFor(branch) ? 0 : UsedStateBytesFor(branch);
}

std::uint64_t Llm::SpilledStateBytes(const Branch& branch) const {
  CheckBranch(branch);
  return SpilledBytesFor(branch);
}

std::uint64_t Llm::RetainedSpillBytes(const Branch& branch) const {
  CheckBranch(branch);
  return RetainedSpillBytesFor(branch);
}

Status Llm::ReleaseRetainedSpill(Branch& branch) {
  CheckBranch(branch);
  return ReleaseRetainedSpillFor(branch);
}

Clock::time_point Llm::LastUsed(const Branch& branch) const {
  CheckBranch(branch);
  return branch.history_used_.at;
}

memory::ReclaimStamp Llm::LastStamp(const Branch& branch) const {
  CheckBranch(branch);
  return branch.history_used_.reclaim;
}

std::uint64_t Llm::SpillWriteBytes(const Branch& branch) const {
  CheckBranch(branch);
  return SpillWriteBytesFor(branch);
}

std::size_t Llm::ReusablePrefix(const Branch& branch, std::span<const std::int32_t> tokens) const {
  CheckBranch(branch);
  const auto now = Clock::now();
  if (branch.needs_clear_ || branch.history_.empty() || !StateUsableFor(branch) ||
      (!branch.HeldByContinuation() && now - branch.history_used_.at >= retention_)) {
    return 0;
  }
  const std::size_t common = CommonPrefix(branch.history_, tokens);
  if (common == branch.history_.size() && common < tokens.size()) {
    return common;
  }
  std::vector<TurnBoundary> boundaries;
  boundaries.reserve(branch.turn_checkpoints_.size());
  for (const TurnCheckpoint& checkpoint : branch.turn_checkpoints_) {
    boundaries.push_back(checkpoint.boundary);
  }
  const auto match = MatchingTurnBoundary(boundaries, common, tokens.size(), now, retention_);
  return match ? boundaries[*match].position : 0;
}

// ------------------------------------------- conversations kept (D-105)

void Llm::set_kept(Kept kept) {
  kept_ = std::move(kept);
  if (kept_.keeper == nullptr) {
    SetStageSpillPlaces({});
    return;
  }
  // Each slot's spill files named beneath the model's private directory
  // (one a stage), kept as they are for the slots whose records validated.
  SetStageSpillPlaces([dir = kept_.directory, adopt = kept_.adopt, files = kept_.files,
                       spill = checkpoint_directory_](std::uint32_t slot, std::uint32_t stage) {
    const auto at = std::ranges::find(adopt, slot);
    const auto index = static_cast<std::size_t>(at - adopt.begin());
    // A stage's file is kept only as the one its record names, which
    // the start checked: without that identity it starts fresh.
    std::optional<platform::FileIdentity> expect;
    if (at != adopt.end() && index < files.size() && stage < files[index].size()) {
      expect = serving_support::FileIdentityOf(files[index][stage]);
    }
    return engine::LiveState::SpillPlace{.directory = spill,
                                         .dir = dir,
                                         .name = kept::StateFileName(slot, stage),
                                         .keep = expect.has_value(),
                                         .expect = expect};
  });
}

void Llm::Unkeep(const Branch& branch) {
  if (kept_.keeper != nullptr) {
    kept_.keeper->Invalidate(kept_.model, BranchIndex(branch));
  }
}

bool Llm::SettleIdle(Branch& branch) {
  if (!OwedFor(branch)) {
    return true;
  }
  if (node_->InRequest(paged().stream()) || !BranchIdle(branch) || SpilledFor(branch) ||
      !StateUsableFor(branch)) {
    return false;
  }
  // Selected alone (no request holds the stream), its owed restore runs as
  // the next job's would have: the same bytes, now.
  std::array<Branch*, 1> alone{&branch};
  if (auto selected = SelectBranches(alone); !selected) {
    return false;
  }
  return SettleFor(branch).has_value();
}

void Llm::KeepBranch(Branch& branch) {
  if (kept_.keeper == nullptr || branch.history_.empty() || branch.needs_clear_ ||
      !branch.score_rows_.bytes.empty() || !KeptWholeFor(branch)) {
    return;
  }
  // One file a stage, each of its stage's one region.
  const std::vector<const engine::LiveState*> lives = KeptLivesFor(branch);
  if (lives.empty() || std::ranges::any_of(lives, [&](const engine::LiveState* live) {
        return live == nullptr || !live->spill_identity() || live->regions() != 1;
      })) {
    return;
  }
  MemoryCharge token_charge;
  kept::Record r;
  if (!ReserveTokens(branch, r.tokens, token_charge, branch.history_.size())) {
    Unkeep(branch);
    Say(std::format("{}: a conversation record was not kept: token history memory refused",
                    name()));
    return;
  }
  r.version = kept::kVersion;
  r.identity = kept_.identity;
  r.slot = BranchIndex(branch);
  for (std::size_t f = 0; f < lives.size(); ++f) {
    const engine::LiveState& live = *lives[f];
    kept::StateFile file{.name = kept::StateFileName(r.slot, static_cast<std::uint32_t>(f)),
                         .id = KeptId(*live.spill_identity()),
                         .bytes = 0};
    for (std::size_t i = 0; i < live.regions(); ++i) {
      r.regions.push_back(live.mapped_bytes(i));
      file.bytes += live.mapped_bytes(i);
    }
    r.files.push_back(std::move(file));
    // An extent's region is its stage.
    for (const engine::LiveState::Range& range : live.used_ranges()) {
      r.extents.push_back({.region = static_cast<std::uint32_t>(f),
                           .index = static_cast<std::uint32_t>(range.offset / kExtent),
                           .digest = {}});
    }
  }
  r.tokens = branch.history_;
  r.cursor = CursorFor(branch);
  const auto decoding = TurnDecodingStateFor(branch).Save();
  r.decoding.assign(decoding.begin(), decoding.end());
  r.used_unix_ms = UnixMs(branch.history_used_.at);
  for (const TurnCheckpoint& c : branch.turn_checkpoints_) {
    if (!c.file.identity() || c.file.name().empty() || c.boundary.position > r.tokens.size()) {
      continue;
    }
    const auto words = c.decoding.Save();
    r.checkpoints.push_back({.file = c.file.name(),
                             .id = KeptId(*c.file.identity()),
                             .file_bytes = c.file.file_bytes(),
                             .position = static_cast<std::uint32_t>(c.boundary.position),
                             .created_unix_ms = UnixMs(c.boundary.created),
                             .ranges = KeptRanges(c.file.ranges()),
                             .footprint = KeptRanges(c.footprint),
                             .cursor = c.cursor,
                             .decoding = {words.begin(), words.end()},
                             .digests = {}});
  }
  kept_.keeper->Keep(kept_.model, std::move(r), std::move(token_charge));
}

Status Llm::Adopt(Branch& branch, kept::Record& record, MemoryCharge& token_charge) {
  CheckIdleGeneration(branch);
  const HistoryFundingGuard funding(branch.funding_history_);
  if (KeptLivesFor(branch).empty() || kept_.keeper == nullptr) {
    return Error("this model keeps no conversation across a restart");
  }
  // The slot's turn checkpoint files the branch does not hold (left out,
  // or all of them when nothing was adopted) go: nothing stays unused.
  const std::uint32_t slot = BranchIndex(branch);
  const auto remove_unused_checkpoints = [&]() {
    auto entries = platform::ListPrivateDirectory(kept_.directory);
    if (!entries) {
      return;
    }
    for (const platform::DirectoryEntry& entry : *entries) {
      const auto named = kept::ParseFileName(entry.name);
      if (named && named->kind == kept::NameOf::Kind::kCheckpoint && named->slot == slot &&
          std::ranges::none_of(branch.turn_checkpoints_, [&](const TurnCheckpoint& t) {
            return t.file.name() == entry.name;
          })) {
        std::ignore = platform::RemovePrivate(kept_.directory, entry.name.c_str());
      }
    }
  };
  // A record not adopted goes, so no later start tries it again.
  const auto refused = [&](std::string why) -> Status {
    const std::string name = kept::RecordFileName(slot);
    std::ignore = platform::RemovePrivate(kept_.directory, name.c_str());
    remove_unused_checkpoints();
    return Error(std::move(why));
  };
  // The draft depth's state is the conversation's; its relative cost is the
  // machine's, calibrated (D-103) and perhaps measured again since: the
  // model's own now, never a reason to refuse.
  const double cost = BranchDecoding(branch).relative_cost();
  const auto same = [&](const execution::AdaptiveDepth& depth) {
    return depth.maximum() == BranchDecoding(branch).maximum();
  };
  const auto loaded = execution::AdaptiveDepth::Load(record.decoding);
  if (!loaded.has_value() || !same(*loaded)) {
    return refused("its adaptive draft depth is not this model's");
  }
  const execution::AdaptiveDepth decoding = loaded->WithCost(cost);
  if (record.version != kept::kVersion) {
    return refused("its format version is not this model's");
  }
  // The slot's spill files, reopened since the start checked them, are the
  // very ones the record names (a stage each): one put in a name's place
  // since was not used (LiveState::RegisterSpill), and is refused here.
  const std::vector<const engine::LiveState*> lives = KeptLivesFor(branch);
  if (lives.size() != record.files.size()) {
    return refused("its spill files are not this model's");
  }
  for (std::size_t f = 0; f < lives.size(); ++f) {
    if (lives[f] == nullptr || !lives[f]->spill_identity() ||
        KeptId(*lives[f]->spill_identity()) != record.files[f].id) {
      return refused(
          std::format("its spill file {} is not the one its record names", record.files[f].name));
    }
  }
  // Its extents as the slot's used ranges (version 2: `region` the stage),
  // each bounded to its layout (kept::Check held each inside it).
  const std::vector<std::uint64_t> layouts = KeptLayouts();
  std::vector<engine::LiveState::Range> used;
  used.reserve(record.extents.size());
  for (const kept::Extent& extent : record.extents) {
    const std::uint64_t offset = std::uint64_t{extent.index} * kExtent;
    if (extent.region >= layouts.size() || offset >= layouts[extent.region]) {
      return refused("an extent lies outside this model's layout");
    }
    used.push_back({.region = extent.region,
                    .offset = offset,
                    .bytes = std::min(kExtent, layouts[extent.region] - offset)});
  }
  // Its checkpoints, each opened (one that does not is left out).
  std::vector<TurnCheckpoint> checkpoints;
  for (kept::Checkpoint& c : record.checkpoints) {
    const auto depth = execution::AdaptiveDepth::Load(c.decoding);
    if (!depth.has_value() || !same(*depth)) {
      Say(
          std::format("{}: a kept turn checkpoint at {} tokens was left out: its draft depth is "
                      "not this model's",
                      name(), c.position));
      continue;
    }
    if (auto checked =
            CheckCheckpointMetadataFor(branch, c.position, c.cursor, LiveRanges(c.footprint));
        !checked) {
      Say(std::format("{}: a kept turn checkpoint at {} was left out: {}", name(), c.position,
                      checked.error()));
      continue;
    }
    auto file = engine::CheckpointFile::Adopt(kept_.directory, c.file, LiveRanges(c.ranges),
                                              serving_support::FileIdentityOf(c.id));
    if (!file) {
      Say(std::format("{}: a kept turn checkpoint at {} tokens was left out: {}", name(),
                      c.position, file.error()));
      continue;
    }
    checkpoints.push_back(
        {.boundary = {.position = c.position, .created = SteadyAt(c.created_unix_ms)},
         .file = std::move(*file),
         .footprint = LiveRanges(c.footprint),
         .cursor = c.cursor,
         .decoding = depth->WithCost(cost),
         .layout = record.identity.layout});
    if (const auto named = kept::ParseFileName(c.file); named) {
      checkpoint_serial_ = std::max(checkpoint_serial_, named->serial + 1);
    }
  }
  if (auto adopted =
          AdoptPositionFor(branch, record.tokens, record.cursor, used, record.identity.layout);
      !adopted) {
    return refused(adopted.error());
  }
  ReleaseTokens(branch.history_, branch.history_charge_);
  branch.history_ = std::move(record.tokens);
  branch.history_charge_ = std::move(token_charge);
  branch.needs_clear_ = false;
  branch.history_used_ = SteadyAt(record.used_unix_ms);
  RestoreTurnDecodingStateFor(branch, decoding);
  branch.turn_checkpoints_ = std::move(checkpoints);
  remove_unused_checkpoints();
  // The record stays: it describes the slot's file as it is.
  std::erase_if(record.checkpoints, [&](const kept::Checkpoint& c) {
    return std::ranges::none_of(branch.turn_checkpoints_,
                                [&](const TurnCheckpoint& t) { return t.file.name() == c.file; });
  });
  kept_.keeper->Adopted(kept_.model, record);
  return {};
}

void Llm::ForgetDiscarded(std::uint32_t discarded) {
  for (std::uint32_t slot = 0; slot < branch_count_; ++slot) {
    if ((discarded & (std::uint32_t{1} << slot)) != 0) {
      Branch& branch = slot == 0 ? default_branch_ : *extra_branches_[slot - 1];
      Forget(branch);
      branch.needs_clear_ = false;  // its runner discarded it already
    }
  }
}

std::vector<std::string> Llm::UnkeptSlots() const {
  std::vector<std::string> out;
  if (kept_.keeper == nullptr) {
    return out;
  }
  for (std::uint32_t slot = 0; slot < branch_count_; ++slot) {
    const Branch& branch = slot == 0 ? default_branch_ : *extra_branches_[slot - 1];
    for (const engine::LiveState* live : KeptLivesFor(branch)) {
      if (live != nullptr && !live->spill_note().empty()) {
        out.push_back(std::format("slot {}: {}", slot, live->spill_note()));
        break;
      }
    }
  }
  return out;
}

void Llm::PreserveKeptFiles() {
  const auto preserve = [](Branch& branch) {
    for (TurnCheckpoint& checkpoint : branch.turn_checkpoints_) {
      checkpoint.file.Preserve();
    }
  };
  preserve(default_branch_);
  for (std::uint32_t slot = 1; slot < branch_count_; ++slot) {
    preserve(*extra_branches_[slot - 1]);
  }
}

Status Llm::SpillIdle(Branch& branch) { return Spill(branch, false); }

Status Llm::SpillSetAside(Branch& branch) { return Spill(branch, true); }

Status Llm::Spill(Branch& branch, bool set_aside) {
  CheckIdleGeneration(branch);
  if (!set_aside && !BranchIdle(branch)) {
    return Error("a conversation in use is not spilled");
  }
  if (SpilledFor(branch)) {
    return {};
  }
  // State that may hold anything (a failed step's), or none, is dropped:
  // there is nothing exact to keep. A member set aside is cleared by its
  // caller, within its request (Branch::Clear).
  if (branch.needs_clear_ || branch.history_.empty() || !StateUsableFor(branch)) {
    return set_aside ? Error("no exact state to spill") : ReleaseIdleState(branch);
  }
  // Kept across a restart (D-105): its last verify's owed restore runs
  // first where nothing holds the stream, so the file holds it whole.
  if (!set_aside && kept_.keeper != nullptr) {
    (void)SettleIdle(branch);
  }
  // What it writes (only what changed since its last spill): the measured
  // rate's bytes.
  const std::uint64_t bytes = SpillWriteBytesFor(branch);
  const auto started = Clock::now();
  if (auto spilled = SpillFor(branch); !spilled) {
    if (set_aside || branch.HeldByContinuation()) {
      return spilled;
    }
    // Neither spilled nor known as it was: dropped, its next turn prefills.
    if (auto released = ReleaseIdleState(branch); !released) {
      return Error(std::format("{}; clearing it instead: {}", spilled.error(), released.error()));
    }
    return Error(std::format("{}: cleared instead", spilled.error()));
  }
  ++spill_stats_.spills;
  spill_stats_.spilled_bytes += bytes;
  spill_stats_.spill_seconds += Seconds(Clock::now() - started);
  KeepBranch(branch);  // its record follows, in the background
  return {};
}

Status Llm::RestoreSpilled(Branch& branch, bool& refused) {
  refused = false;
  if (!SpilledFor(branch)) {
    return {};
  }
  const std::uint64_t bytes = SpilledBytesFor(branch);
  const auto started = Clock::now();
  if (auto restored = RestoreFor(branch); !restored) {
    refused = StateRefusedFor(branch) && StateUsableFor(branch);
    // Refused for room, it is still wholly spilled and its record (D-105)
    // stands for a restart; anything else may have changed its files.
    if (!refused || !SpilledFor(branch)) {
      Unkeep(branch);
    }
    return restored;
  }
  // Live again: its file may change from here on (D-105).
  Unkeep(branch);
  ++spill_stats_.restores;
  spill_stats_.restored_bytes += bytes;
  spill_stats_.restore_seconds += Seconds(Clock::now() - started);
  return {};
}

bool Llm::CapacityRefused(const Branch& branch) const {
  return StateRefusedFor(branch) && StateUsableFor(branch);
}

bool Llm::ReclaimFor(const Branch& branch) {
  return capacity_reclaim_ && capacity_reclaim_(branch);
}

bool Llm::sampling(const Branch& branch) const {
  CheckBranch(branch);
  return branch.sampling_.has_value();
}

execution::AdaptiveDepth& Llm::BranchDecoding(Branch& branch) {
  CheckBranch(branch);
  return branch.decoding_;
}

const execution::AdaptiveDepth& Llm::BranchDecoding(const Branch& branch) const {
  CheckBranch(branch);
  return branch.decoding_;
}

void Llm::SaveBranchDecoding(Branch& branch) {
  CheckBranch(branch);
  branch.saved_decoding_ = branch.decoding_;
}

void Llm::RestoreBranchDecoding(Branch& branch) {
  CheckBranch(branch);
  branch.decoding_ = branch.saved_decoding_;
}

void Llm::CheckIdleGeneration(const Branch& branch, const PromptSession* prompt) const {
  CheckBranch(branch);
  base::Check(!branch.generation_active_, "mutating a conversation with an active generation");
  base::Check(branch.prompt_session_ == prompt,
              "mutating a conversation with another active prompt");
}

void Llm::Forget(Branch& branch, const PromptSession* prompt) {
  CheckIdleGeneration(branch, prompt);
  Unkeep(branch);
  branch.turn_checkpoints_.clear();
  branch.score_rows_ = {};
  ReleaseTokens(branch.history_, branch.history_charge_);
  branch.needs_clear_ = true;
}

Status Llm::Clear(Branch& branch, const PromptSession* prompt) {
  CheckIdleGeneration(branch, prompt);
  Unkeep(branch);
  branch.turn_checkpoints_.clear();
  if (auto r = ClearStateFor(branch); !r) {
    return r;
  }
  branch.score_rows_ = {};
  ReleaseTokens(branch.history_, branch.history_charge_);
  branch.needs_clear_ = false;
  branch.history_used_ = Clock::now();
  return {};
}

Status Llm::ReleaseIdleState(Branch& branch) {
  if (branch.HeldByContinuation()) {
    return Error("an in-flight continuation holds this conversation's state");
  }
  CheckIdleGeneration(branch);
  Unkeep(branch);
  branch.turn_checkpoints_.clear();
  branch.score_rows_ = {};
  ReleaseTokens(branch.history_, branch.history_charge_);
  if (auto released = ReleaseIdleStateFor(branch); !released) {
    branch.needs_clear_ = true;  // its next use clears whatever is left
    return released;
  }
  branch.needs_clear_ = false;
  branch.history_used_ = Clock::now();
  return {};
}

std::size_t Llm::ExpireTurnCheckpoints(Branch& branch, Clock::time_point now) {
  if (!BranchIdle(branch) || branch.HeldByContinuation()) {
    return 0;
  }
  const std::size_t expired = std::erase_if(branch.turn_checkpoints_, [&](const TurnCheckpoint& c) {
    return now - c.boundary.created >= retention_;
  });
  // Their files went with them; a kept record names them no more.
  if (expired != 0 && kept_.keeper != nullptr &&
      kept_.keeper->Kept(kept_.model, BranchIndex(branch))) {
    KeepBranch(branch);
  }
  return expired;
}

std::uint64_t Llm::Branch::turn_checkpoint_bytes() const {
  std::uint64_t bytes = checkpoint_capture_bytes_;
  for (const TurnCheckpoint& checkpoint : turn_checkpoints_) {
    const auto file_bytes = checkpoint.file.file_bytes();
    if (file_bytes > std::numeric_limits<std::uint64_t>::max() - bytes)
      return std::numeric_limits<std::uint64_t>::max();
    bytes += file_bytes;
  }
  return bytes;
}

Status Llm::CaptureTurnCheckpoint(Branch& branch, const PrefillGoOn& go_on, bool& stopped,
                                  const PromptSession* prompt) {
  if (branch.history_.empty()) {
    return {};
  }
  const std::size_t position = branch.history_.size();
  if (std::ranges::any_of(branch.turn_checkpoints_, [&](const TurnCheckpoint& c) {
        return c.boundary.position == position;
      })) {
    return {};  // looking it up does not renew its retention period
  }
  if (auto settled = SettleFor(branch); !settled) {
    return settled;
  }
  auto ranges = CheckpointRangesFor(branch, static_cast<std::uint32_t>(position));
  if (!ranges) {
    return std::unexpected(ranges.error());
  }
  if (!turn_checkpoints_) {
    return {};  // the speed gate's diagnostic: no save, as the M3 Gemma classes'
  }
  std::uint64_t file_bytes = 0;
  for (const auto& range : *ranges) {
    constexpr auto alignment = kept::kFileAlignment;
    const auto most = std::numeric_limits<std::uint64_t>::max();
    if (range.bytes > most - (alignment - 1)) return {};
    const auto padded = (range.bytes + alignment - 1) / alignment * alignment;
    if (padded > most - file_bytes) return {};
    file_bytes += padded;
  }
  if (checkpoint_admission_ && !checkpoint_admission_(file_bytes)) {
    return {};  // the disk budget has no room for this optional cache
  }
  // Staging can synchronously reclaim idle state after the admission.
  // Hold its disk capacity in the branch's accounting throughout capture;
  // release it after a failed file is destroyed, or transfer it to the
  // published file without invoking any intervening callback.
  base::Check(branch.checkpoint_capture_bytes_ == 0, "a checkpoint capture is already under way");
  struct CaptureCapacity {
    std::uint64_t& bytes;
    ~CaptureCapacity() { bytes = 0; }
  } capacity{branch.checkpoint_capture_bytes_};
  capacity.bytes = file_bytes;
  const engine::CheckpointFile::Continue progress =
      go_on ? engine::CheckpointFile::Continue([&]() { return go_on(0); })
            : engine::CheckpointFile::Continue{};
  // Named, beneath the model's private directory, when conversations are
  // kept across a restart (D-105): a record may name it.
  engine::CheckpointFile::Place place{.directory = checkpoint_directory_, .dir = -1, .name = {}};
  if (kept_.keeper != nullptr) {
    place.dir = kept_.directory;
    place.name = kept::CheckpointFileName(BranchIndex(branch), checkpoint_serial_++);
  }
  auto file = engine::CheckpointFile::Capture(
      *node_, place, *ranges,
      [&](void* host, std::span<const engine::LiveState::Range> page) {
        return CopyCheckpointStateFor(branch, host, page, true);
      },
      progress);
  if (!file) {
    if (file.error().invalid_state) {
      Forget(branch, prompt);
      return Error("saving a turn checkpoint: " + file.error().detail);
    }
    stopped = file.error().cancelled;
    if (!stopped) {
      // Positions only (D-014): the next turn that would continue from it
      // prefills instead.
      Say(std::format("{}: a turn checkpoint at {} tokens was not saved: {}", name(), position,
                      file.error().detail));
    }
    return {};  // optional cache: unavailable staging or disk does not fail the turn
  }
  TurnCheckpoint checkpoint{.boundary = {.position = position, .created = Clock::now()},
                            .file = std::move(*file),
                            .footprint = UsedStateRangesFor(branch),
                            .cursor = CursorFor(branch),
                            .decoding = TurnDecodingStateFor(branch),
                            .layout = KeptLayout()};
  branch.turn_checkpoints_.push_back(std::move(checkpoint));
  capacity.bytes = 0;  // its retained file now carries the charge
  // The kept record's format bound; the disk's is the spill budget's
  // (Server::TrimTurnCheckpoints, between units).
  if (branch.turn_checkpoints_.size() > kept::kMostCheckpoints) {
    branch.turn_checkpoints_.erase(branch.turn_checkpoints_.begin());
  }
  return {};
}

std::optional<Clock::time_point> Llm::OldestTurnCheckpoint(const Branch& branch) const {
  CheckBranch(branch);
  if (branch.turn_checkpoints_.empty() || branch.checkpoint_capture_bytes_ != 0 ||
      !BranchIdle(branch) || branch.HeldByContinuation()) {
    return std::nullopt;
  }
  return std::ranges::min_element(branch.turn_checkpoints_, {},
                                  [](const TurnCheckpoint& c) { return c.boundary.created; })
      ->boundary.created;
}

std::uint64_t Llm::DropOldestTurnCheckpoint(Branch& branch) {
  if (!OldestTurnCheckpoint(branch)) {
    return 0;
  }
  const auto oldest = std::ranges::min_element(
      branch.turn_checkpoints_, {}, [](const TurnCheckpoint& c) { return c.boundary.created; });
  const std::uint64_t bytes = oldest->file.file_bytes();
  branch.turn_checkpoints_.erase(oldest);  // its file with it
  // A kept record names it no more.
  if (kept_.keeper != nullptr && kept_.keeper->Kept(kept_.model, BranchIndex(branch))) {
    KeepBranch(branch);
  }
  return bytes;
}

std::uint32_t Llm::ReusableFor(const Branch& branch, std::span<const std::int32_t> tokens,
                               bool fresh, bool resume) const {
  if (fresh) {
    return 0;
  }
  const std::size_t prefix = ReusablePrefix(branch, tokens);
  // A generation set aside for its peers resumes from its whole prompt.
  if (prefix == 0 && resume && !branch.history_.empty() &&
      branch.history_.size() == tokens.size() && !branch.needs_clear_ && StateUsableFor(branch) &&
      CommonPrefix(branch.history_, tokens) == tokens.size()) {
    return static_cast<std::uint32_t>(tokens.size());
  }
  return static_cast<std::uint32_t>(prefix);
}

Status Llm::ReusePrompt(Branch& branch, std::span<const std::int32_t> tokens, std::uint32_t& reused,
                        bool fresh, bool resume, const PrefillGoOn& go_on, bool& stopped,
                        const PromptSession* prompt) {
  reused = 0;
  const auto now = Clock::now();
  if (fresh || branch.needs_clear_ || !StateUsableFor(branch) ||
      (!branch.HeldByContinuation() && now - branch.history_used_.at >= retention_)) {
    return Clear(branch, prompt);
  }
  std::erase_if(branch.turn_checkpoints_, [&](const TurnCheckpoint& checkpoint) {
    return now - checkpoint.boundary.created >= retention_;
  });
  const std::size_t common = CommonPrefix(branch.history_, tokens);
  // The live state continues the prompt (with `resume`, it may be the
  // whole prompt: a generation set aside for its peers resumes from it).
  const bool live = !branch.history_.empty() && common == branch.history_.size() &&
                    (common < tokens.size() || (resume && common == tokens.size()));
  std::vector<TurnBoundary> boundaries;
  boundaries.reserve(branch.turn_checkpoints_.size());
  for (const TurnCheckpoint& checkpoint : branch.turn_checkpoints_) {
    boundaries.push_back(checkpoint.boundary);
  }
  const auto match = live
                         ? std::nullopt
                         : MatchingTurnBoundary(boundaries, common, tokens.size(), now, retention_);
  if (!live && !match) {
    return Clear(branch, prompt);  // a spilled state is discarded with it
  }
  // Spill, not clear: an idle conversation's state spilled for capacity
  // comes back exactly as it left before it is reused (refused for
  // capacity: the unit runs again once the runtime made room).
  if (bool refused = false; SpilledFor(branch)) {
    if (auto restored = RestoreSpilled(branch, refused); !restored) {
      if (refused) {
        return restored;  // the caller defers it (PromptSession::Advance)
      }
      Say(
          std::format("{}: a spilled conversation of {} tokens was not restored ({}): the "
                      "prompt is prefilled from its start",
                      name(), branch.history_.size(), restored.error()));
      return Clear(branch, prompt);
    }
  }
  if (live) {
    reused = static_cast<std::uint32_t>(common);
    branch.history_used_ = now;
    return {};
  }
  TurnCheckpoint& checkpoint = branch.turn_checkpoints_[*match];
  const auto position = static_cast<std::uint32_t>(checkpoint.boundary.position);
  const auto restored_cursor = checkpoint.cursor;
  const auto restored_decoding = checkpoint.decoding;
  if (auto settled = SettleFor(branch); !settled) {
    Forget(branch, prompt);
    return settled;
  }
  const engine::CheckpointFile::Continue progress =
      go_on ? engine::CheckpointFile::Continue([&]() { return go_on(0); })
            : engine::CheckpointFile::Continue{};
  auto restored = checkpoint.file.Restore(
      *node_,
      [&]() {
        return PreparePositionRestoreFor(branch, position, restored_cursor, checkpoint.footprint,
                                         checkpoint.layout);
      },
      [&](void* host, std::span<const engine::LiveState::Range> page) {
        return CopyCheckpointStateFor(branch, host, page, false);
      },
      progress);
  if (!restored) {
    if (StateRefusedFor(branch) && StateUsableFor(branch)) return Error(restored.error().detail);
    if (restored.error().cancelled) {
      stopped = true;
      return restored.error().invalid_state ? Clear(branch, prompt) : Status{};
    }
    Say(
        std::format("{}: the turn checkpoint at {} tokens was not restored ({}): the prompt is "
                    "prefilled from its start",
                    name(), position, restored.error().detail));
    // Before mutation this is an optional cache miss. After mutation, Clear
    // drops the branch before any fresh prefill; an uncertain device copy
    // may make Clear fail, in which case the runtime stops normally.
    return Clear(branch, prompt);
  }
  if (auto completed = CompletePositionRestoreFor(branch, position); !completed) {
    Forget(branch, prompt);
    return completed;
  }
  branch.history_.resize(position);
  RestoreTurnDecodingStateFor(branch, restored_decoding);
  std::erase_if(branch.turn_checkpoints_,
                [&](const TurnCheckpoint& c) { return c.boundary.position > position; });
  reused = static_cast<std::uint32_t>(position);
  branch.needs_clear_ = false;
  branch.history_used_ = now;
  return {};
}

Status Llm::SaveState(Branch& branch, void* host) {
  auto identity = ExecutionIdentity();
  if (!identity) return Error(identity.error());
  if (*identity) return Error("this source requires an opaque snapshot capture");
  CheckIdleGeneration(branch);
  const HistoryFundingGuard funding(branch.funding_history_);
  if (!ReserveTokens(branch, branch.saved_history_, branch.saved_history_charge_,
                     branch.history_.size())) {
    return Error("native snapshot history exceeds the execution budget");
  }
  branch.saved_valid_ = false;
  if (auto r = SettleFor(branch); !r) {
    branch.InvalidateStateSnapshot();
    return r;
  }
  auto ranges = UsedStateRangesFor(branch);
  if (auto r = SaveUsedStateFor(branch, host, ranges); !r) {
    branch.InvalidateStateSnapshot();
    return r;
  }
  branch.saved_ranges_ = std::move(ranges);
  branch.saved_history_ = branch.history_;
  if (branch.saved_history_.empty()) {
    ReleaseTokens(branch.saved_history_, branch.saved_history_charge_);
  }
  branch.saved_cursor_ = CursorFor(branch);
  branch.saved_score_rows_ = branch.score_rows_;
  SaveDecodingStateFor(branch);
  branch.saved_valid_ = true;
  return {};
}

Status Llm::RestoreState(Branch& branch, void* host) {
  auto identity = ExecutionIdentity();
  if (!identity) return Error(identity.error());
  if (*identity) return Error("this source requires an authenticated snapshot capability");
  CheckIdleGeneration(branch);
  const HistoryFundingGuard funding(branch.funding_history_);
  if (!branch.saved_valid_) {
    return Error("no completed conversation snapshot to restore");
  }
  if (host == nullptr && !branch.saved_ranges_.empty()) {
    return Error("the conversation snapshot has no source buffer");
  }
  if (!ReserveTokens(branch, branch.history_, branch.history_charge_,
                     branch.saved_history_.size())) {
    return Error("native restored history exceeds the execution budget");
  }
  Unkeep(branch);
  if (auto r = SettleFor(branch); !r) {
    return r;
  }
  if (auto r = RestoreSnapshotFor(branch, host, branch.saved_ranges_, branch.saved_cursor_,
                                  KeptLayout());
      !r) {
    if (StateRefusedFor(branch) && StateUsableFor(branch)) return r;
    branch.needs_clear_ = true;
    ReleaseTokens(branch.history_, branch.history_charge_);
    return r;
  }
  branch.history_ = branch.saved_history_;
  branch.score_rows_ = branch.saved_score_rows_;
  branch.turn_checkpoints_.clear();  // a full diagnostic restore may replace the branch
  branch.history_used_ = Clock::now();
  RestoreDecodingStateFor(branch);
  branch.needs_clear_ = false;
  return {};
}

std::expected<StateSnapshot, std::string> Llm::Branch::CaptureState(SnapshotStore& store) {
  return model_.CaptureState(*this, store);
}

Status Llm::Branch::RestoreState(SnapshotStore& store, const StateSnapshot& snapshot) {
  return model_.RestoreState(*this, store, snapshot);
}

std::expected<StateSnapshot, std::string> Llm::CaptureState(Branch& branch, SnapshotStore& store) {
  auto identity = ExecutionIdentity();
  if (!identity) return std::unexpected(identity.error());
  if (auto checked = store.CheckNode(node_); !checked) return std::unexpected(checked.error());
  CheckIdleGeneration(branch);
  const HistoryFundingGuard funding(branch.funding_history_);
  // Opaque captures never allocate branch.saved_history_: the single
  // snapshot allocation funds history and all other captured metadata.
  if (auto settled = SettleFor(branch); !settled) return std::unexpected(settled.error());
  const auto ranges = UsedStateRangesFor(branch);
  if (branch.history_.size() > std::numeric_limits<std::uint32_t>::max()) {
    return std::unexpected("the snapshot history exceeds its position representation");
  }
  if (auto metadata = CheckCheckpointMetadataFor(
          branch, static_cast<std::uint32_t>(branch.history_.size()), CursorFor(branch), ranges);
      !metadata) {
    return std::unexpected(metadata.error());
  }
  if (auto prepared =
          store.Prepare(branch.snapshot_issuer_, *identity, KeptLayout(), CursorFor(branch), ranges,
                        branch.history_, TurnDecodingStateFor(branch));
      !prepared) {
    return std::unexpected(prepared.error());
  }
  if (auto saved = SaveUsedStateFor(branch, store.WriteAddress(), ranges); !saved) {
    store.Unproven();
    return std::unexpected(saved.error());
  }
  auto snapshot = store.Publish();
  snapshot.score_rows_ = branch.score_rows_;
  return snapshot;
}

Status Llm::RestoreState(Branch& branch, SnapshotStore& store, const StateSnapshot& snapshot) {
  // All capability/issuer/identity checks precede funding, reads of payload,
  // owed-state settlement, kept-file mutation and any device submission.
  auto identity = ExecutionIdentity();
  if (!identity) return Error(identity.error());
  if (auto checked = store.CheckNode(node_); !checked) return checked;
  auto captured = store.Read(snapshot, branch.snapshot_issuer_, *identity, KeptLayout());
  if (!captured) return Error(captured.error());
  if (captured->history.size() > std::numeric_limits<std::uint32_t>::max()) {
    return Error("the captured history exceeds its position representation");
  }
  if (auto metadata =
          CheckCheckpointMetadataFor(branch, static_cast<std::uint32_t>(captured->history.size()),
                                     captured->cursor, captured->ranges);
      !metadata) {
    return metadata;
  }
  CheckIdleGeneration(branch);
  const HistoryFundingGuard funding(branch.funding_history_);
  if (!ReserveTokens(branch, branch.history_, branch.history_charge_, captured->history.size())) {
    return Error("native restored history exceeds the execution budget");
  }
  Unkeep(branch);
  if (auto settled = SettleFor(branch); !settled) return settled;
  if (auto restored = RestoreSnapshotFor(branch, captured->payload, captured->ranges,
                                         captured->cursor, captured->layout);
      !restored) {
    if (StateRefusedFor(branch) && StateUsableFor(branch)) return restored;
    store.Unproven();
    branch.needs_clear_ = true;
    ReleaseTokens(branch.history_, branch.history_charge_);
    return restored;
  }
  branch.history_.assign(captured->history.begin(), captured->history.end());
  branch.score_rows_ = snapshot.score_rows_;
  branch.turn_checkpoints_.clear();
  branch.history_used_ = Clock::now();
  RestoreTurnDecodingStateFor(branch, captured->decoding);
  branch.needs_clear_ = false;
  return {};
}

}  // namespace llmp::runtime
