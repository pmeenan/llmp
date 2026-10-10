// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// An LLM's conversations on the node (serving.h's Llm): its branches,
// prompt and generation sessions, turn checkpoints, spills and kept
// conversations (D-105), over the virtual calls each model implements.
// Vendor-free: it reaches the engine through the core (the paged node,
// live state, checkpoint files), so it builds in every profile, with the
// engine-backed adapter (engine_llm.h) and its tests on the fake backend.

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

class HistoryFundingGuard {
 public:
  explicit HistoryFundingGuard(bool& funding)
      : funding_(funding), before_(std::exchange(funding, true)) {}
  HistoryFundingGuard(const HistoryFundingGuard&) = delete;
  HistoryFundingGuard& operator=(const HistoryFundingGuard&) = delete;
  HistoryFundingGuard(HistoryFundingGuard&&) = delete;
  HistoryFundingGuard& operator=(HistoryFundingGuard&&) = delete;
  ~HistoryFundingGuard() { funding_ = before_; }

 private:
  bool& funding_;
  bool before_;
};

// The highest value's index, the first of equals (greedy choice).
std::int32_t HostArgmax(std::span<const float> row) {
  return static_cast<std::int32_t>(std::ranges::max_element(row) - row.begin());
}

}  // namespace

// ---------------------------------------------------------------- an LLM

Llm::Llm() : default_branch_(*this) {}

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
    bytes += b.history_charge_.bytes() + b.saved_history_charge_.bytes() +
             b.verify_tokens_charge_.bytes();
  }
  return bytes;
}

std::uint64_t Llm::IdleHistoryBytes(const Branch& branch) const {
  CheckBranch(branch);
  return branch.history_charge_.bytes() + branch.verify_tokens_charge_.bytes();
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

bool Llm::ReserveTokenScratch(Branch& branch, std::size_t tokens, std::uint32_t left) {
  return ReserveTokens(branch, branch.verify_tokens_, branch.verify_tokens_charge_,
                       TokenScratchCapacityFor(branch, tokens, left));
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
  saved_valid_ = false;
  model_.ReleaseTokens(saved_history_, saved_history_charge_);
  saved_ranges_.clear();
}

Llm::Branch::~Branch() {
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
                                  bool fresh) {
  return model_.PreparePrompt(*this, tokens, stable_boundary, last, reused, go_on, run, fresh);
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
                          PrefillRun* run, bool fresh) {
  return default_branch_.PreparePrompt(tokens, stable_boundary, last, reused, go_on, run, fresh);
}

std::uint64_t Llm::turn_checkpoint_bytes() const { return default_branch_.turn_checkpoint_bytes(); }

std::expected<std::unique_ptr<Llm::GenerationSession>, std::string> Llm::BeginGeneration(
    const std::vector<float>& last, const GenerateOptions& options, Generation& out) & {
  return default_branch_.BeginGeneration(last, options, out);
}

Status Llm::Generate(const std::vector<float>& last, const GenerateOptions& options,
                     Generation& out) {
  return default_branch_.Generate(last, options, out);
}

std::uint64_t Llm::state_snapshot_bytes() const { return default_branch_.state_snapshot_bytes(); }

Status Llm::SaveState(void* host) { return default_branch_.SaveState(host); }

Status Llm::RestoreState(void* host) { return default_branch_.RestoreState(host); }

std::expected<std::vector<std::int32_t>, std::string> Llm::EncodeText(std::string_view text,
                                                                      RequestMemory* memory,
                                                                      std::uint64_t* needed) const {
  std::vector<tokenizer::TokenId> ids;
  // The working set and the tokens (a byte each at most, grown by
  // doubling), charged while they are built.
  MemoryCharge charge;
  if (memory != nullptr) {
    const std::uint64_t need = tokenizer_->WorkingBytes(text) +
                               (std::uint64_t{2} * sizeof(tokenizer::TokenId) * (text.size() + 2));
    if (!charge.Add(*memory, need)) {
      if (needed != nullptr) {
        *needed = need;
      }
      return Error(std::format("encoding the text needs {} bytes of request memory", need));
    }
  }
  // Bounded by the text itself (D-102): every token covers at least a
  // byte, beside a BOS and an EOS.
  if (auto r = tokenizer_->Encode(text,
                                  {.special = tokenizer::SpecialTokens::kUserDefinedOnly,
                                   .add_bos_eos = true,
                                   .max_bytes = text.size(),
                                   .max_tokens = text.size() + 2},
                                  ids);
      !r) {
    return Error(r.error().ToString());
  }
  if (tokenizer_->bos() && (ids.empty() || ids.front() != *tokenizer_->bos())) {
    ids.insert(ids.begin(), *tokenizer_->bos());  // BOS first, as the references fed it
  }
  return ids;
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
  // The tokenization's working set and its tokens (a byte each at most,
  // grown by doubling), charged while they are built.
  MemoryCharge charge;
  if (options.memory != nullptr) {
    const std::uint64_t window = tokenizer_->WorkingBytes(rendered->text);
    const std::uint64_t need =
        window + (std::uint64_t{2} * sizeof(tokenizer::TokenId) *
                  std::min<std::uint64_t>(rendered->text.size(), options.max_tokens));
    if (need > options.memory->capacity()) {
      // A stretch without a cut point too long to tokenize here at all,
      // whatever else the node holds: the prompt's shape, not the moment.
      if (failure != nullptr) {
        *failure = ChatRenderFailure::kUnbroken;
      }
      if (options.memory_needed != nullptr) {
        *options.memory_needed = window / tokenizer::kEncodeBytesPerWindowByte;
      }
      return Error("the conversation holds text too long without a break to tokenize");
    }
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

std::expected<std::int32_t, std::string> Llm::Choose(std::span<const float> row,
                                                     std::uint64_t position) {
  return Choose(default_branch_, row, position);
}

std::expected<std::int32_t, std::string> Llm::Choose(Branch& branch, std::span<const float> row,
                                                     std::uint64_t position) {
  CheckBranch(branch);
  if (!branch.sampling_) {
    return HostArgmax(row);
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

std::expected<bool, std::string> Llm::Keep(std::span<const float> row, std::int32_t draft,
                                           std::uint64_t position, std::int32_t& next) {
  return Keep(default_branch_, row, draft, position, next);
}

std::expected<bool, std::string> Llm::Keep(Branch& branch, std::span<const float> row,
                                           std::int32_t draft, std::uint64_t position,
                                           std::int32_t& next) {
  CheckBranch(branch);
  if (!branch.sampling_) {
    next = HostArgmax(row);
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

Status Llm::UseTemplate(std::string_view text) {
  // A rendering's bounds follow the model (D-102): what a prompt that fits
  // its context could occupy; a value and the values held at once scale
  // with it, never below the library's own. The longest token's bytes: a
  // normal token's decoded, a special token's text.
  const std::size_t longest = tokenizer_->longest_token_bytes();
  longest_token_ = longest;
  render_bytes_ = static_cast<std::size_t>(RenderBytes(
      context_, longest, tokenizer_->normalization() == tokenizer::Normalization::kNfc));
  chat::jinja::Limits limits;
  limits.max_output_bytes = render_bytes_;
  limits.max_string_bytes = std::max(limits.max_string_bytes, render_bytes_);
  limits.max_live_bytes = std::max(limits.max_live_bytes, 2 * render_bytes_);
  auto chosen = chat::ChatTemplate::ForText(text, chat::TokenFacts::From(*tokenizer_), limits);
  if (!chosen) {
    return Error(chosen.error());
  }
  template_ = std::move(*chosen);
  auto stops = chat::StopTokens(template_->stop(), *tokenizer_);
  if (!stops) {
    return Error(stops.error().ToString());
  }
  stops_.assign(stops->begin(), stops->end());
  if (tokenizer_->eos() && std::ranges::find(stops_, *tokenizer_->eos()) == stops_.end()) {
    stops_.push_back(*tokenizer_->eos());
  }
  if (template_->how() != chat::ChatTemplate::How::kNativeByHash) {
    std::println(stderr, "{}: chat template {} rendered {} (D-067)", name_,
                 template_->sha256().substr(0, 12),
                 template_->how() == chat::ChatTemplate::How::kNativeByProbe
                     ? std::format("natively by {}, equal on its probe corpus", template_->name())
                     : std::string("by the sandboxed template interpreter"));
  }
  return {};
}

Status Llm::UseChatAssets(const ja::Artifact& artifact, const config::ModelEntry& entry) {
  auto assets = ReadChatAssets(artifact, entry);
  if (!assets) {
    return std::unexpected(assets.error());
  }
  auto created = tokenizer::Tokenizer::Create(std::move(assets->spec));
  if (!created) {
    return Error(std::format("{}: {}", assets->from, created.error().ToString()));
  }
  tokenizer_ = std::make_unique<tokenizer::Tokenizer>(std::move(*created));
  // The reasoning markers (D-103): an override, else the vocabulary's
  // "<think>" pair, else none.
  if (auto r = ResolveReasoning(settings_, entry,
                                [this](std::string_view text) { return tokenizer_->Find(text); });
      !r) {
    return r;
  }
  const auto find = [this](const std::string& text) -> std::optional<std::int32_t> {
    return text.empty() ? std::nullopt : tokenizer_->Find(text);
  };
  think_start_ = find(settings_.reasoning_start.value);
  think_end_ = find(settings_.reasoning_end.value);
  // A model is refused when it could serve no turn.
  if (!assets->chat_template) {
    return Error(
        std::format("{} keeps no chat template and models.{}.chat_template is not set, "
                    "so it has no chat turns",
                    assets->from, entry.name));
  }
  return UseTemplate(*assets->chat_template);
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

Status Llm::SelectBranches(std::span<Branch* const> active) {
  if (active.size() != 1 || active.front() != &default_branch_) {
    return Error("this model supports only its default native conversation");
  }
  return {};
}

void Llm::CheckBranch(const Branch& branch) const {
  base::Check(&branch.model_ == this && branch.slot_ < branch_count_ &&
                  (branch.slot_ == 0 ? &branch == &default_branch_
                                     : &branch == extra_branches_[branch.slot_ - 1].get()),
              "conversation branch does not belong to this model's native slot");
}

void Llm::CheckDefaultBranch(const Branch& branch) const {
  CheckBranch(branch);
  base::Check(&branch == &default_branch_, "this family forwards native work only for branch zero");
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
  // (version 2: one a stage), kept as they are for the slots whose records
  // validated.
  SetStageSpillPlaces(
      [dir = kept_.directory, adopt = kept_.adopt, files = kept_.files,
       spill = checkpoint_directory_,
       one_file = KeptVersion() == kept::kOneFileVersion](std::uint32_t slot, std::uint32_t stage) {
        const auto at = std::ranges::find(adopt, slot);
        const auto index = static_cast<std::size_t>(at - adopt.begin());
        // A stage's file is kept only as the one its record names, which
        // the start checked: without that identity it starts fresh.
        std::optional<platform::FileIdentity> expect;
        if (at != adopt.end() && index < files.size() && stage < files[index].size()) {
          expect = serving_support::FileIdentityOf(files[index][stage]);
        }
        return engine::LiveState::SpillPlace{
            .directory = spill,
            .dir = dir,
            .name = one_file ? kept::StateFileName(slot) : kept::StateFileName(slot, stage),
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
      !KeptWholeFor(branch)) {
    return;
  }
  // Version 1: one file of every region; version 2: one a stage, each of
  // its stage's one region.
  const std::uint32_t version = KeptVersion();
  const std::vector<const engine::LiveState*> lives = KeptLivesFor(branch);
  if (lives.empty() || (version == kept::kOneFileVersion && lives.size() != 1) ||
      std::ranges::any_of(lives, [&](const engine::LiveState* live) {
        return live == nullptr || !live->spill_identity() ||
               (version != kept::kOneFileVersion && live->regions() != 1);
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
  r.version = version;
  r.identity = kept_.identity;
  r.slot = BranchIndex(branch);
  for (std::size_t f = 0; f < lives.size(); ++f) {
    const engine::LiveState& live = *lives[f];
    kept::StateFile file{.name = version == kept::kOneFileVersion
                                     ? kept::StateFileName(r.slot)
                                     : kept::StateFileName(r.slot, static_cast<std::uint32_t>(f)),
                         .id = KeptId(*live.spill_identity()),
                         .bytes = 0};
    for (std::size_t i = 0; i < live.regions(); ++i) {
      r.regions.push_back(live.mapped_bytes(i));
      file.bytes += live.mapped_bytes(i);
    }
    r.files.push_back(std::move(file));
    // Version 1's ranges name the file's region; version 2's the stage.
    for (const engine::LiveState::Range& range : live.used_ranges()) {
      r.extents.push_back({.region = static_cast<std::uint32_t>(
                               version == kept::kOneFileVersion ? range.region : f),
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
  if (record.version != KeptVersion()) {
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
         .decoding = depth->WithCost(cost)});
    if (const auto named = kept::ParseFileName(c.file); named) {
      checkpoint_serial_ = std::max(checkpoint_serial_, named->serial + 1);
    }
  }
  if (auto adopted = AdoptPositionFor(branch, record.tokens, record.cursor, used); !adopted) {
    return refused(adopted.error());
  }
  ReleaseTokens(branch.history_, branch.history_charge_);
  branch.history_ = std::move(record.tokens);
  branch.history_charge_ = std::move(token_charge);
  branch.needs_clear_ = false;
  branch.history_used_ = SteadyAt(record.used_unix_ms);
  SetCursorFor(branch, record.cursor);
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
  // Live again: its file may change from here on (D-105).
  Unkeep(branch);
  const std::uint64_t bytes = SpilledBytesFor(branch);
  const auto started = Clock::now();
  if (auto restored = RestoreFor(branch); !restored) {
    refused = StateRefusedFor(branch) && StateUsableFor(branch);
    return restored;
  }
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

Status Llm::RunChunkFor(Branch& branch, std::span<const std::int32_t> all, std::uint32_t n_past,
                        bool inject, std::vector<float>& logits) {
  CheckDefaultBranch(branch);
  return RunChunk(all, n_past, inject, logits);
}

Status Llm::RunPrefillChunkFor(Branch& branch, std::span<const std::int32_t> all,
                               std::uint32_t n_past, bool inject, bool /*want_head*/,
                               std::vector<float>& logits, PrefillHint /*next*/) {
  return RunChunkFor(branch, all, n_past, inject, logits);
}

Status Llm::SpecStepFor(Branch& branch, std::span<const std::int32_t> all, std::uint32_t pos,
                        std::uint32_t left, std::vector<std::int32_t>& kept,
                        std::vector<std::vector<float>>* logits, std::uint64_t& drafted,
                        bool* /*prefix_kept*/) {
  CheckDefaultBranch(branch);
  return SpecStep(all, pos, left, kept, logits, drafted);
}

Status Llm::SettleFor(Branch& branch) {
  CheckDefaultBranch(branch);
  return Settle();
}

Status Llm::ClearStateFor(Branch& branch) {
  CheckDefaultBranch(branch);
  return ClearState();
}

bool Llm::StateUsableFor(const Branch& branch) const {
  CheckDefaultBranch(branch);
  return StateUsable();
}

Status Llm::PrepareDecodeStateFor(Branch& branch, std::uint32_t pos, std::uint32_t left) {
  CheckDefaultBranch(branch);
  return PrepareDecodeState(pos, left);
}

std::uint64_t Llm::TargetStateBaseFor(const Branch& branch) const {
  CheckDefaultBranch(branch);
  return target_state_base();
}

std::uint64_t Llm::TargetStateBytesFor(const Branch& branch) const {
  CheckDefaultBranch(branch);
  return target_state_bytes();
}

std::uint64_t Llm::DrafterStateBaseFor(const Branch& branch) const {
  CheckDefaultBranch(branch);
  return drafter_state_base();
}

std::uint64_t Llm::DrafterStateBytesFor(const Branch& branch) const {
  CheckDefaultBranch(branch);
  return drafter_state_bytes();
}

std::uint64_t Llm::UsedStateBytesFor(const Branch& branch) const {
  CheckDefaultBranch(branch);
  return used_state_bytes();
}

std::vector<engine::LiveState::Range> Llm::UsedStateRangesFor(const Branch& branch) const {
  CheckDefaultBranch(branch);
  return used_state_ranges();
}

Status Llm::SaveUsedStateFor(Branch& branch, void* host,
                             std::span<const engine::LiveState::Range> ranges) {
  CheckDefaultBranch(branch);
  return SaveUsedState(host, ranges);
}

Status Llm::RestoreUsedStateFor(Branch& branch, void* host,
                                std::span<const engine::LiveState::Range> ranges) {
  CheckDefaultBranch(branch);
  return RestoreUsedState(host, ranges);
}

std::expected<std::vector<engine::LiveState::Range>, std::string> Llm::CheckpointRangesFor(
    const Branch& branch, std::uint32_t positions) const {
  CheckDefaultBranch(branch);
  return CheckpointRanges(positions);
}

Status Llm::PrepareRestoreStateFor(Branch& branch,
                                   std::span<const engine::LiveState::Range> footprint) {
  CheckDefaultBranch(branch);
  return PrepareRestoreState(footprint);
}

Status Llm::CopyCheckpointStateFor(Branch& branch, void* host,
                                   std::span<const engine::LiveState::Range> ranges, bool to_host) {
  CheckDefaultBranch(branch);
  return CopyCheckpointState(host, ranges, to_host);
}

std::uint32_t Llm::CursorFor(const Branch& branch) const {
  CheckDefaultBranch(branch);
  return cursor();
}

void Llm::SetCursorFor(Branch& branch, std::uint32_t value) {
  CheckDefaultBranch(branch);
  set_cursor(value);
}

void Llm::SaveDecodingStateFor(Branch& branch) {
  CheckDefaultBranch(branch);
  SaveDecodingState();
}

void Llm::RestoreDecodingStateFor(Branch& branch) {
  CheckDefaultBranch(branch);
  RestoreDecodingState();
}

execution::AdaptiveDepth Llm::TurnDecodingStateFor(const Branch& branch) const {
  CheckDefaultBranch(branch);
  return TurnDecodingState();
}

void Llm::RestoreTurnDecodingStateFor(Branch& branch, const execution::AdaptiveDepth& state) {
  CheckDefaultBranch(branch);
  RestoreTurnDecodingState(state);
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
  ReleaseTokens(branch.history_, branch.history_charge_);
  ReleaseTokens(branch.verify_tokens_, branch.verify_tokens_charge_);
  branch.needs_clear_ = true;
}

Status Llm::Clear(Branch& branch, const PromptSession* prompt) {
  CheckIdleGeneration(branch, prompt);
  Unkeep(branch);
  branch.turn_checkpoints_.clear();
  if (auto r = ClearStateFor(branch); !r) {
    return r;
  }
  ReleaseTokens(branch.history_, branch.history_charge_);
  ReleaseTokens(branch.verify_tokens_, branch.verify_tokens_charge_);
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
  ReleaseTokens(branch.history_, branch.history_charge_);
  ReleaseTokens(branch.verify_tokens_, branch.verify_tokens_charge_);
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
  std::uint64_t bytes = 0;
  for (const TurnCheckpoint& checkpoint : turn_checkpoints_) {
    bytes += checkpoint.file.bytes();
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
                            .decoding = TurnDecodingStateFor(branch)};
  branch.turn_checkpoints_.push_back(std::move(checkpoint));
  if (branch.turn_checkpoints_.size() > kTurnCheckpointLimit) {
    branch.turn_checkpoints_.erase(branch.turn_checkpoints_.begin());
  }
  return {};
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
        return PreparePositionRestoreFor(branch, position, restored_cursor, checkpoint.footprint);
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
  SetCursorFor(branch, restored_cursor);
  RestoreTurnDecodingStateFor(branch, restored_decoding);
  std::erase_if(branch.turn_checkpoints_,
                [&](const TurnCheckpoint& c) { return c.boundary.position > position; });
  reused = static_cast<std::uint32_t>(position);
  branch.needs_clear_ = false;
  branch.history_used_ = now;
  return {};
}

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
  if (max_rows_ == 0) {
    return Error("a prompt needs a positive prefill chunk size");
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
    unit.rows = scoring_ ? 1 : PrefillRows(end - at, model_.max_rows_);
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
  if (checkpoint_pending_ && at == stable_boundary_) {
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
         a->want_head == b->want_head &&
         model_.CompatiblePrefill(static_cast<std::uint32_t>(branch_.history_.size()), a->rows,
                                  static_cast<std::uint32_t>(peer.branch_.history_.size()),
                                  b->rows);
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
    if (unit->want_head == want_head &&
        anchor.model_.CompatiblePrefill(static_cast<std::uint32_t>(anchor.branch_.history_.size()),
                                        unit->rows, at, rows))
      return i;
  }
  return chosen;
}

PrefillHint Llm::PromptSession::NextPrefillHint(std::uint32_t current_end) const {
  PrefillHint hint;
  const auto boundary =
      checkpoint_pending_ ? stable_boundary_ : static_cast<std::uint32_t>(tokens_.size());
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
                                         bool defer_capacity) {
  const auto at = static_cast<std::uint32_t>(branch_.history_.size());
  const auto end = at + rows;
  if (!result) {
    std::string error =
        std::format("{}'s prefill: the chunk at {}: {}", model_.name_, at, result.error());
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
  if (scoring_ && end < tokens_.size() && on_row_ && !on_row_(tokens_[end], last_)) Stop();
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
    return std::unexpected(next.error());
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
    auto reused =
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
  } else {
    const auto at = static_cast<std::uint32_t>(branch_.history_.size());
    const auto end = at + next->rows;
    if (auto prepared = PrepareChunk(next->rows, defer_capacity); !prepared) return prepared;
    const auto hint = NextPrefillHint(end);
    const auto started = Clock::now();
    const StateKeeper::Quiet quiet(model_.kept_.keeper);  // records' hashing waits
    auto chunk =
        model_.RunPrefillChunkFor(branch_, std::span(tokens_).first(end), at, model_.speculate_,
                                  scoring_ || end == tokens_.size(), last_, hint);
    const double seconds = Seconds(Clock::now() - started);
    if (auto completed_chunk =
            CompleteChunk(next->rows, seconds, seconds, std::move(chunk), defer_capacity);
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
  if (!branch.history_.empty() || tokens.empty() || tokens.size() > context_) {
    return Error("literal scoring needs an empty history and a nonempty prompt within context");
  }
  if (branch.needs_clear_) {
    if (auto r = Clear(branch); !r) {
      return r;
    }
  }
  PrefillRun completed;
  last.clear();
  for (std::uint32_t at = 0; at < tokens.size(); ++at) {
    if (go_on && !go_on(1)) {
      completed.stopped = true;
      break;
    }
    if (!ReserveTokens(branch, branch.history_, branch.history_charge_, at + 1)) {
      branch.capacity_refused_ = true;
      return Error("native scoring history exceeds the execution budget");
    }
    const auto started = Clock::now();
    auto ran = RunChunkFor(branch, tokens.first(std::size_t{at} + 1), at, speculate_, last);
    // A capacity refusal ran nothing: the same row again once freed.
    for (std::size_t reclaimed = 0;
         !ran && CapacityRefused(branch) && reclaimed < kMaxBranches && ReclaimFor(branch);
         ++reclaimed) {
      ran = RunChunkFor(branch, tokens.first(std::size_t{at} + 1), at, speculate_, last);
    }
    if (!ran) {
      branch.capacity_refused_ = CapacityRefused(branch);
      branch.needs_clear_ = !StateUsableFor(branch);
      if (branch.needs_clear_) {
        ReleaseTokens(branch.history_, branch.history_charge_);
        ReleaseTokens(branch.verify_tokens_, branch.verify_tokens_charge_);
      }
      last.clear();
      return Error(std::format("{}'s scoring: the row at {}: {}", name_, at, ran.error()));
    }
    branch.history_.push_back(tokens[at]);
    completed.end = at + 1;
    ++completed.chunks;
    completed.longest = std::max(completed.longest, Seconds(Clock::now() - started));
    if (at + 1 < tokens.size() && on_row && !on_row(tokens[at + 1], last)) {
      completed.stopped = true;
      break;
    }
  }
  if (auto settled = SettleFor(branch); !settled) {
    branch.needs_clear_ = true;
    ReleaseTokens(branch.history_, branch.history_charge_);
    ReleaseTokens(branch.verify_tokens_, branch.verify_tokens_charge_);
    last.clear();
    return settled;
  }
  branch.history_used_ = Clock::now();
  if (completed.stopped) {
    last.clear();
  }
  if (run != nullptr) {
    *run = completed;
  }
  return {};
}

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
  if (auto began = session->Begin(last, resume); !began) {
    session->Close();
    return std::unexpected(began.error());
  }
  return session;
}

bool Llm::GenerationSession::IsStop(std::int32_t token) const {
  return options_.stop &&
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
  if (out_.stopped && visible == out_.tokens.size() && visible > 0) {
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
    return {};
  }
  auto first = model_.Choose(branch_, last, branch_.history_.size());
  if (!first) {
    return std::unexpected(first.error());
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
  const auto step_tokens = model_.speculate_ ? model_.StepTokenBoundFor(branch_, left_) : 1U;
  if (!model_.ReserveTokens(branch_, all_, all_charge_, all_.size() + step_tokens) ||
      !model_.ReserveTokenScratch(branch_, all_.size(), left_)) {
    refusal_ = "native generation history exceeds the execution budget";
    if (defer_capacity) {
      return std::unexpected(refusal_);
    }
    ran_ = Error(refusal_);
    failed_prefix_valid_ = true;
    return std::unexpected(ran_.error());
  }
  if (auto prepared = model_.PrepareDecodeStateFor(branch_, position_, left_); !prepared) {
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
              .need_logits = options_.keep_logits || static_cast<bool>(options_.on_logits)};
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
  ++position_;  // the anchor was processed before the choice failed
  failed_prefix_valid_ = model_.StateUsableFor(branch_);
  return ran_;
}

bool Llm::GreedyWithoutRows(const PreparedGeneration& unit) {
  return unit.branch != nullptr && unit.session != nullptr && !unit.step.need_logits &&
         !unit.branch->sampling_.has_value();
}

bool Llm::DeviceGreedy(const PreparedGeneration& unit) {
  return !unit.step.speculative && GreedyWithoutRows(unit);
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
                                           std::vector<std::vector<float>> logits) {
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
    const std::int32_t token = kept[i];
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
    if (options_.on_logits) {
      out_.cancelled = !Report() || out_.cancelled;
    }
    if (out_.stopped || out_.cancelled) {
      break;
    }
  }
  if (!options_.on_logits) {
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
  if (step->speculative) {
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
  if (!step->need_logits && !branch_.sampling_) {
    std::int32_t token = 0;
    if (auto ran = model_.RunGreedyChunkFor(branch_, step->all, step->position, token)) {
      if (!*ran) {
        return FailStep(ran->error());
      }
      model_.calibration_samples_.DecodeStep(step->position, 1.0, Seconds(Clock::now() - started));
      return ApplyChosen(token);
    }
  }
  std::vector<float> row;
  if (auto ran = model_.RunChunkFor(branch_, step->all, step->position, false, row); !ran) {
    return FailStep(ran.error());
  }
  model_.calibration_samples_.DecodeStep(step->position, 1.0, Seconds(Clock::now() - started));
  return ApplyPlain(std::move(row));
}

Status Llm::RunPreparedGenerationWave(std::span<PreparedGeneration> prepared) {
  // Families without a shared implementation retain the ordinary one-slot
  // unit. This fallback never advertises multi-request shared dispatch.
  if (prepared.size() != 1) {
    return Error("this model has no shared generation implementation");
  }
  PreparedGeneration& unit = prepared.front();
  if (unit.step.speculative) {
    return SpecStepFor(*unit.branch, unit.step.all, unit.step.position, unit.step.left, unit.kept,
                       unit.step.need_logits ? &unit.logits : nullptr, unit.drafted,
                       &unit.failed_prefix_valid);
  }
  return RunPlainGenerationUnit(unit);
}

Status Llm::RunPlainGenerationUnit(PreparedGeneration& unit) {
  if (unit.step.speculative) return Error("a plain generation unit cannot be speculative");
  unit.chosen.reset();
  if (DeviceGreedy(unit)) {
    std::int32_t token = 0;
    if (auto ran = RunGreedyChunkFor(*unit.branch, unit.step.all, unit.step.position, token)) {
      if (*ran) unit.chosen = token;
      return *ran;
    }
  }
  return RunChunkFor(*unit.branch, unit.step.all, unit.step.position, false, unit.row);
}

Status Llm::RunScalarGenerationUnits(std::span<PreparedGeneration> prepared) {
  for (auto& unit : prepared) {
    if (unit.step.speculative) return Error("independent scalar units require plain generation");
    unit.result = RunPlainGenerationUnit(unit);
    if (!unit.result && !GenerationCohortUsable()) return unit.result;
    if (!unit.result) unit.failed_prefix_valid = StateUsableFor(*unit.branch);
  }
  return {};
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
        tokens += unit.step.speculative ? unit.kept.size() : 1;
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
      (void)(unit.step.speculative ? unit.session->ApplySpeculative(
                                         std::move(unit.kept), std::move(unit.logits), unit.drafted)
             : unit.chosen         ? unit.session->ApplyChosen(*unit.chosen)
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
  branch_.sampling_.reset();
  model_.RetireSamplingScratchFor(branch_);
  model_.ReleaseTokens(all_, all_charge_);
  model_.ReleaseTokens(branch_.verify_tokens_, branch_.verify_tokens_charge_);
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
                     Generation& out) {
  CheckBranch(branch);
  branch.capacity_refused_ = false;
  auto began = BeginGeneration(branch, last, options, out);
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

Status Llm::SaveState(Branch& branch, void* host) {
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
  SaveDecodingStateFor(branch);
  branch.saved_valid_ = true;
  return {};
}

Status Llm::RestoreState(Branch& branch, void* host) {
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
  if (auto r = RestoreSnapshotFor(branch, host, branch.saved_ranges_, branch.saved_cursor_); !r) {
    if (StateRefusedFor(branch) && StateUsableFor(branch)) return r;
    branch.needs_clear_ = true;
    ReleaseTokens(branch.history_, branch.history_charge_);
    return r;
  }
  branch.history_ = branch.saved_history_;
  branch.turn_checkpoints_.clear();  // a full diagnostic restore may replace the branch
  branch.history_used_ = Clock::now();
  SetCursorFor(branch, branch.saved_cursor_);
  RestoreDecodingStateFor(branch);
  branch.needs_clear_ = false;
  return {};
}

}  // namespace llmp::runtime
