// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <format>
#include <utility>

#include "base/check.h"
#include "engine/draft_companions.h"
#include "model/state_contract.h"
#include "runtime/model_settings.h"
#include "runtime/serving.h"
#include "runtime/serving_support.h"

namespace llmp::runtime {
namespace {

namespace en = llmp::engine;
namespace pl = llmp::planner;
using serving_support::Error;

std::int32_t Argmax(std::span<const float> row) {
  return static_cast<std::int32_t>(std::ranges::max_element(row) - row.begin());
}

}  // namespace

Llm::Llm(en::PagedNode& node, const config::ModelEntry& entry, const ModelSettings& settings,
         const config::RuntimeRoles& roles, int index, LlmBuild build)
    : default_branch_(*this),
      node_ref_(node),
      entry_(entry),
      store_(roles.installed),
      spill_(roles.spill),
      index_(index),
      build_(std::move(build)) {
  name_ = entry.name;
  settings_ = settings;
  node_ = &node;
  context_ = settings.context.value;
  max_rows_ = std::max<std::uint32_t>(settings.prefill_chunk.value, 1);
  configured_rows_ = settings.prefill_chunk.value;
  checkpoint_directory_ = roles.spill;
  slots_ = std::clamp<std::uint32_t>(settings.max_slots.value, 1,
                                     static_cast<std::uint32_t>(kMaxBranches));
}

Llm::~Llm() = default;

Status Llm::Built() const {
  if (!runner_ || !composer_ || !plain_) return Error(name_ + " is not set up");
  if (auto identity = ExecutionIdentity(); !identity) return Error(identity.error());
  return {};
}

// ------------------------------------------------------------------ its life

en::PagedModel& Llm::paged() {
  base::Check(static_cast<bool>(runner_), "an engine-backed model's runner before its Setup");
  return *runner_;
}

Status Llm::Setup() {
  if (static_cast<bool>(runner_)) return Error(std::format("model {}: set up already", name_));
  if (!build_) return Error(std::format("model {}: no component pipeline builds it", name_));
  auto parts = build_(settings_);
  if (!parts) return Error(std::format("model {}: {}", name_, parts.error()));
  if (parts->logit_width == 0) return Error(std::format("model {}: its head has no logits", name_));
  // Its tokenizer and chat template, from its artifact (D-067); a model
  // without one (the tests') has no chat turns.
  if (entry_.artifact) {
    std::optional<artifact::Artifact> artifact;
    if (parts->target_artifact && parts->target_artifact->id() == *entry_.artifact) {
      artifact = std::move(parts->target_artifact);
    } else {
      auto opened = OpenInstalled(store_, *entry_.artifact);
      if (!opened) return Error(opened.error());
      artifact.emplace(std::move(*opened));
    }
    if (auto used = UseChatAssets(*artifact, entry_, parts->startup_cap); !used) return used;
  }
  parts->target_artifact.reset();
  logit_width_ = parts->logit_width;
  caps_ = parts->caps;
  prefill_ = parts->prefill;
  device_greedy_ = parts->device_greedy;
  device_verify_ = parts->device_verify;
  branch_depth_ = parts->branch_depth;
  if (branch_depth_) depth_cost_ratio_ = settings_.depth_cost_ratio.value;
  drafter_specs_ = parts->mode.drafters;
  wave_mode_ = std::move(parts->wave_mode);
  if (branch_depth_ > 7 || (branch_depth_ && drafter_specs_.size() > 1))
    return Error("a conversation-owned depth requires exactly one drafter");
  architecture_ = parts->architecture;
  chat_ = std::move(parts->chat);
  keep_ = std::move(parts->keep);
  startup_host_bytes_ = parts->startup_host_bytes;
  execution_metadata_ = std::move(parts->execution_metadata);
  // Control tokens that end a chat reply (a channel the slice does not
  // parse), where the tokenizer has them.
  chat_stops_.clear();
  if (tokenizer_ != nullptr) {
    for (const std::string& text : chat_.stops) {
      if (const auto token = tokenizer_->Find(text)) chat_stops_.push_back(*token);
    }
  }
  en::SlotLifecycleOptions& slots = parts->config.slots;
  slots.model = name_;
  slots.slots = slots_;
  if (slots.spill_directory.empty()) slots.spill_directory = spill_;
  // Named files kept across a restart once the server says where (D-105:
  // set before Register), else unnamed ones.
  slots.spill_place = [this, directory = slots.spill_directory](std::uint32_t slot,
                                                                std::uint32_t stage) {
    return stage_place_ ? stage_place_(slot, stage)
                        : en::LiveState::SpillPlace{
                              .directory = directory, .dir = -1, .name = {}, .keep = false};
  };
  slots.content_changed = [this, also = std::move(slots.content_changed)](std::uint32_t slot) {
    if (slot < reset_.size()) reset_[slot] = true;
    if (also) also(slot);
  };
  const bool drafters =
      parts->mode.kind == en::DecodingMode::Kind::kSpeculative && !parts->mode.drafters.empty();
  // Speculation on names a drafter (the settings', D-103): a model whose
  // parts have none refuses rather than serving plain in its place.
  if (settings_.speculation.value && !drafters) {
    return Error(std::format("model {}: speculation is on but its parts have no drafter", name_));
  }
  const bool speculate = drafters && settings_.speculation.value;
  context_margin_ = 0;
  if (speculate) {
    for (const auto& drafter : drafter_specs_)
      if (drafter.draft_before_limit)
        context_margin_ = std::max(context_margin_, drafter.max_depth);
  }
  // A complete draft pass needs this tail even when the reply ends sooner.
  // Reserve the configured depth, independently of the adaptive policy ceiling.
  const auto effective_context =
      slots.stages.empty() ? context_ : std::min(context_, slots.stages.front().layout.context);
  if (context_margin_ != 0 && context_margin_ >= effective_context)
    return Error("the complete draft schedule does not fit the target context");
  verify_rows_ = speculate ? parts->mode.verify_rows() : 1;
  if (plan_report_sink_) parts->config.plan_report_sink = plan_report_sink_;
  auto runner =
      en::ModelRunner::Create(node_ref_, std::move(parts->config), std::move(parts->source), index_,
                              static_cast<std::uint32_t>(index_));
  if (!runner) return Error(std::format("model {}: {}", name_, runner.error()));
  auto retained = en::RunnerOwner::Adopt(node_ref_, *runner, keep_);
  if (!retained) return Error(retained.error());
  runner_ = std::move(*retained);
  if (auto set = runner_->Setup(); !set) return set;
  auto identity = runner_->ExecutionIdentity();
  if (!identity) return Error(identity.error());
  execution_identity_ = *identity;
  // The settings' context and chunk, as the target's layout holds them.
  const model::StateLayoutDerived& layout = runner_->slots().layout(0);
  context_ = std::min(context_, layout.context);
  max_rows_ = std::max<std::uint32_t>(std::min(max_rows_, layout.max_rows), 1);
  if (caps_.max_segment_rows != 0) max_rows_ = std::min(max_rows_, caps_.max_segment_rows);
  auto composer = en::WaveComposer::For(caps_, runner_->slots());
  if (!composer) return Error(std::format("model {}: {}", name_, composer.error()));
  composer_.emplace(std::move(*composer));
  auto plain = en::DecodingOrchestrator::Make(*runner_, *composer_,
                                              {.kind = en::DecodingMode::Kind::kPlain}, {});
  if (!plain) return Error(std::format("model {}: {}", name_, plain.error()));
  plain_.emplace(std::move(*plain));
  if (speculate) {
    auto speculative =
        en::DecodingOrchestrator::Make(*runner_, *composer_, std::move(parts->mode), Policy());
    if (!speculative) return Error(std::format("model {}: {}", name_, speculative.error()));
    speculative_.emplace(std::move(*speculative));
  }
  speculate_ = speculate;
  return PrepareBranches(
      slots_, branch_depth_ ? branch_depth_ : std::clamp<std::uint32_t>(verify_rows_ - 1, 1, 7));
}

Status Llm::Register() {
  if (auto built = Built(); !built) return built;
  return runner_->Register();
}

Status Llm::Bind() {
  if (auto built = Built(); !built) return built;
  return runner_->Bind();
}

std::uint64_t Llm::activations_needed() const {
  return runner_ ? runner_->activations_needed() : 0;
}
std::uint64_t Llm::pool_needed() const { return runner_ ? runner_->pool_needed() : 0; }
std::uint64_t Llm::host_input_bytes() const {
  // Beside the runner's staging: per slot a retained frontier and a
  // prepared row, and the sampling candidates' bound (2 × vocabulary).
  const std::uint64_t heap = std::uint64_t{slots_} * logit_width_ *
                             (2 * sizeof(float) + 2 * sizeof(execution::SamplingCandidate));
  return (runner_ ? runner_->host_input_bytes() : 0) + heap;
}
std::uint64_t Llm::plan_floor_bytes() const { return runner_ ? runner_->plan_floor_bytes() : 0; }

std::string Llm::plan_report() const { return runner_ ? runner_->plan_report() : ""; }
void Llm::set_plan_report_sink(kernels::PlanRecordSink sink) {
  plan_report_sink_ = std::move(sink);
}
void Llm::ReportPlans(const kernels::PlanRecordSink& sink) const {
  if (runner_) runner_->ReportPlans(sink);
}

std::string Llm::slots_report() const {
  return std::format("{} request slot{} ({}: {})", slots_, slots_ == 1 ? "" : "s",
                     SourceName(settings_.max_slots.source), settings_.max_slots.basis);
}

void Llm::ReclaimCandidates(std::uint32_t owner, bool running,
                            std::vector<memory::ReclaimCandidate>& out) {
  if (runner_) runner_->ReclaimCandidates(owner, running, out);
}
std::uint64_t Llm::Reclaim(memory::ReclaimKind kind, std::uint64_t id) {
  return runner_ ? runner_->Reclaim(kind, id) : 0;
}
std::vector<catalog::ExtentId> Llm::weights() const {
  return runner_ ? runner_->weights() : std::vector<catalog::ExtentId>{};
}
std::vector<catalog::ExtentId> Llm::state() const {
  return runner_ ? runner_->slots().state() : std::vector<catalog::ExtentId>{};
}
std::vector<catalog::ExtentId> Llm::kept_state() const {
  return runner_ ? runner_->slots().kept_state() : std::vector<catalog::ExtentId>{};
}
std::vector<catalog::ExtentId> Llm::unchanged_state() const {
  return runner_ ? runner_->slots().unchanged_state() : std::vector<catalog::ExtentId>{};
}
void Llm::StateWrittenBack(bool whole) {
  if (runner_) runner_->slots().StateWrittenBack(whole);
}
const catalog::Closure& Llm::everything() const {
  base::Check(static_cast<bool>(runner_), "an engine-backed model's closure before its Setup");
  return runner_->slots().everything();
}
const catalog::Closure& Llm::request_closure() const {
  base::Check(static_cast<bool>(runner_), "an engine-backed model's closure before its Setup");
  return runner_->slots().execution();
}
Status Llm::PrepareDefaultRequest() {
  std::array<Branch*, 1> selected{&default_branch()};
  return SelectBranches(selected);
}
std::uint64_t Llm::weight_read_bytes() const { return runner_ ? runner_->weight_read_bytes() : 0; }
Status Llm::CheckPlaces() {
  if (auto built = Built(); !built) return built;
  return runner_->slots().CheckPlaces();
}
void Llm::DropPlans() {
  if (runner_) runner_->DropPlans();
}

Status Llm::RecoverInPlace() {
  if (auto built = Built(); !built) return built;
  auto discarded =
      runner_->slots().RecoverInPlace([this](std::uint32_t slot) { UnkeepSlot(slot); });
  if (!discarded) return std::unexpected(discarded.error());
  ForgetDiscarded(*discarded);
  reset_.fill(true);
  // The lifecycle's recovery drops every plan (its drop_plans); again here
  // in case it kept every slot.
  runner_->DropPlans();
  return {};
}

GraphCounts Llm::graphs() const {
  if (!runner_) return {};
  const en::GraphStats& g = runner_->graph_stats();
  return {.eager = g.eager,
          .captured = g.captured,
          .replayed = g.replayed,
          .refused = g.refused,
          .kept = runner_->plans().graphs()};
}

std::string Llm::violations() const {
  return runner_ ? runner_->coverage().first_violation : std::string();
}

std::string Llm::extra() const {
  return std::format(
      R"({{"engine":"component","architecture":"{}","slots":{},"speculative":{},"verify_rows":{},"joined_prefill_steps":{},"joined_prefill_rows":{},"joined_causal_steps":{},"joined_causal_rows":{},"joined_waves":{},"joined_units":{},"speculative_units":{},"drafted":{},"accepted":{},"discarded":{},"greedy_tokens":{},"runner":{}}})",
      architecture_, slots_, speculate_, verify_rows_, counts_.joined_prefill_steps,
      counts_.joined_prefill_rows, counts_.joined_causal_steps, counts_.joined_causal_rows,
      counts_.joined_waves, counts_.joined_units, counts_.speculative_units, counts_.drafted,
      counts_.accepted, counts_.discarded, counts_.greedy_tokens,
      runner_ ? runner_->report_json() : std::string("{}"));
}

// ------------------------------------------------------------------ the LLM

void Llm::Defaults(chat::Conversation& conversation) const {
  if (conversation.enable_thinking) return;
  if (!chat_.thinking) {
    conversation.enable_thinking = false;
  } else if (chat_.default_thinking) {
    conversation.enable_thinking = *chat_.default_thinking;
  }
}

Status Llm::CheckConversation(const chat::Conversation& conversation) const {
  if (!chat_.thinking && conversation.enable_thinking.value_or(false)) {
    return Error(std::format("{}'s thought-channel output is not served yet", name_));
  }
  if (!tool_format() && !conversation.tools.empty()) {
    return Error(std::format("{}'s generated tool calls are not served yet", name_));
  }
  return {};
}

Status Llm::SelectBranches(std::span<Branch* const> active) {
  if (auto built = Built(); !built) return built;
  if (active.size() > slots_) return Error(std::format("too many {} conversation owners", name_));
  std::array<std::uint32_t, en::kMaxRequestSlots> ids{};
  for (std::size_t i = 0; i < active.size(); ++i) {
    if (active[i] == nullptr || &active[i]->model() != this) {
      return Error(std::format("a conversation that is not {}'s", name_));
    }
    ids[i] = Slot(*active[i]);
  }
  active_branches_.assign(active.begin(), active.end());
  return runner_->slots().SelectSlots(std::span(ids).first(active.size()));
}

std::expected<std::uint64_t, std::string> Llm::StateBytesThrough(std::uint32_t positions) const {
  if (!runner_) return std::unexpected(std::format("{} is not built", name_));
  const en::SlotLifecycle& slots = runner_->slots();
  std::uint64_t total = 0;
  for (std::size_t s = 0; s < slots.stages(); ++s) {
    const model::StateLayoutDerived& layout = slots.layout(s);
    auto used =
        model::UsedThrough(layout, std::min(positions, layout.context), slots.read_align(s));
    if (!used) {
      return std::unexpected(std::format("{}'s stage {} state through {} positions: {}", name_, s,
                                         positions, used.error()));
    }
    std::vector<en::LiveState::Range> ranges;
    ranges.reserve(used->size());
    for (const auto& r : *used)
      ranges.push_back({.region = 0, .offset = r.offset, .bytes = r.bytes});
    auto bytes = slots.live(0, s).UsedBytesOf(ranges);
    if (!bytes) {
      return std::unexpected(std::format("{}'s stage {} state through {} positions: {}", name_, s,
                                         positions, bytes.error()));
    }
    total += *bytes;
  }
  return total;
}

std::size_t Llm::generation_wave_capacity() const { return slots_; }

std::size_t Llm::prefill_wave_capacity() const {
  return caps_.join_prefill ? std::min<std::size_t>(slots_, std::max(caps_.max_segments, 1U)) : 1;
}

bool Llm::CompatiblePrefill(std::uint32_t past, std::uint32_t rows, std::uint32_t peer_past,
                            std::uint32_t peer_rows, bool want_head, bool peer_want_head) const {
  return composer_ &&
         composer_->CanJoinPrefill(past, rows, want_head, peer_past, peer_rows, peer_want_head);
}

std::uint32_t Llm::StepTokenBoundFor(const Branch& /*branch*/, std::uint32_t left) const {
  return std::min(left, speculate_ ? verify_rows_ : 1U);
}

// ------------------------------------------------------------- D-105

std::expected<std::optional<base::Sha256Digest>, std::string> Llm::ExecutionIdentity() const {
  if (!runner_) return std::unexpected("the execution source is not set up");
  auto actual = runner_->ExecutionIdentity();
  if (!actual) return std::unexpected(actual.error());
  if (*actual != execution_identity_)
    return std::unexpected("the execution identity changed after setup");
  return execution_identity_;
}

std::string Llm::KeptLayout() const {
  return runner_ ? runner_->slots().kept_layout_id() : std::string();
}

std::vector<std::uint64_t> Llm::KeptRegions() const {
  std::vector<std::uint64_t> out;
  if (!runner_) return out;
  for (std::size_t s = 0; s < runner_->slots().stages(); ++s) {
    out.push_back(runner_->slots().live(0, s).mapped_bytes(0));
  }
  return out;
}

std::vector<std::uint64_t> Llm::KeptLayouts() const {
  std::vector<std::uint64_t> out;
  if (!runner_) return out;
  for (std::size_t s = 0; s < runner_->slots().stages(); ++s) {
    out.push_back(runner_->slots().live(0, s).bytes(0));
  }
  return out;
}

std::vector<const en::LiveState*> Llm::KeptLivesFor(const Branch& branch) const {
  std::vector<const en::LiveState*> out;
  if (!runner_) return out;
  for (std::size_t s = 0; s < runner_->slots().stages(); ++s) {
    out.push_back(&runner_->slots().live(Slot(branch), s));
  }
  return out;
}

bool Llm::KeptWholeFor(const Branch& branch) const {
  return runner_ && runner_->slots().kept_whole(Slot(branch));
}

bool Llm::OwedFor(const Branch& branch) const {
  return runner_ && runner_->slots().owed(Slot(branch));
}

void Llm::SetStageSpillPlaces(const StageSpillPlace& place) { stage_place_ = place; }

// ------------------------------------------------------------- outputs

std::expected<std::size_t, std::string> Llm::HeadRows(std::span<const std::byte> outputs) const {
  const std::size_t row = std::size_t{logit_width_} * sizeof(float);
  if (row == 0 || outputs.empty() || outputs.size() % row != 0) {
    return Error(std::format("{}'s step returned {} output bytes, not rows of {} logits", name_,
                             outputs.size(), logit_width_));
  }
  return outputs.size() / row;
}

std::expected<std::vector<float>, std::string> Llm::LogitsRow(std::span<const std::byte> outputs,
                                                              std::size_t row) const {
  auto rows = HeadRows(outputs);
  if (!rows) return std::unexpected(rows.error());
  if (row >= *rows) return Error(std::format("{}'s step returned no row {}", name_, row));
  std::vector<float> logits(logit_width_);
  std::memcpy(logits.data(), outputs.data() + (row * logits.size() * sizeof(float)),
              logits.size() * sizeof(float));
  return logits;
}

// ------------------------------------------------------------- decoding

en::DecodingPolicy Llm::Policy() {
  en::DecodingPolicy policy;
  // A drafter's proposal a row: its head's most likely token.
  // (A drafter that returns its next features has them after its logits.)
  policy.propose =
      [this](std::uint32_t /*slot*/, std::size_t drafter, std::span<const std::byte> outputs,
             std::uint32_t want) -> std::expected<std::vector<std::int32_t>, std::string> {
    if (drafter >= drafter_specs_.size()) return Error("unknown drafter output mode");
    if (drafter_specs_[drafter].device_tokens) {
      if (want == 0 || outputs.size() < std::size_t{want} * sizeof(std::int32_t))
        return Error("a draft returned fewer native tokens than requested");
      std::vector<std::int32_t> tokens(want);
      std::memcpy(tokens.data(), outputs.data(), tokens.size() * sizeof(tokens.front()));
      for (const auto token : tokens)
        if (token < 0 || std::cmp_greater_equal(token, logit_width_))
          return Error("a draft returned a token outside the target vocabulary");
      return tokens;
    }
    const auto& drafter_spec = drafter_specs_[drafter];
    auto checked_width = en::DrafterHeadRows(drafter_spec, logit_width_);
    if (!checked_width) return Error(checked_width.error());
    const std::size_t width = *checked_width;
    const std::size_t row_bytes = width * sizeof(float);
    const std::size_t rows =
        row_bytes == 0 ? 0 : std::min<std::size_t>(outputs.size() / row_bytes, want);
    if (rows == 0) {
      return Error(std::format("{}'s draft returned {} output bytes, no row of {} logits", name_,
                               outputs.size(), logit_width_));
    }
    std::vector<std::int32_t> proposed;
    for (std::size_t i = 0; i < rows; ++i) {
      const auto row =
          std::span(reinterpret_cast<const float*>(outputs.data() + (i * row_bytes)), width);
      auto token =
          en::DrafterHeadToken(drafter_spec, static_cast<std::uint32_t>(Argmax(row)), logit_width_);
      if (!token) return Error(token.error());
      proposed.push_back(*token);
    }
    return proposed;
  };
  // The verdict: each draft kept while the model keeps it (greedy agreement,
  // or speculative sampling seeded by its position), then the token in place
  // of the first one refused, or after the last.
  policy.judge =
      [this](std::uint32_t slot, std::span<const std::int32_t> verified,
             std::span<const std::byte> outputs) -> std::expected<en::Verdict, std::string> {
    auto branch = this->branch(slot);
    if (!branch) return std::unexpected(branch.error());
    if (verify_tokens_.at(slot)) {
      if (verified.empty() || outputs.size() != verified.size() * sizeof(std::int32_t))
        return Error("a verify returned inconsistent native token counts");
      std::vector<std::int32_t> tokens(verified.size());
      std::memcpy(tokens.data(), outputs.data(), outputs.size());
      for (const auto token : tokens)
        if (token < 0 || std::cmp_greater_equal(token, logit_width_))
          return Error("a verify returned a token outside the target vocabulary");
      std::size_t accepted = 0;
      while (accepted + 1 < verified.size() && tokens[accepted] == verified[accepted + 1])
        ++accepted;
      en::Verdict verdict{.keep = static_cast<std::uint32_t>(accepted + 1), .tokens = {}};
      verdict.tokens.assign(verified.begin() + 1,
                            verified.begin() + 1 + static_cast<std::ptrdiff_t>(accepted));
      verdict.tokens.push_back(tokens[accepted]);
      return verdict;
    }
    auto rows = HeadRows(outputs);
    if (!rows) return std::unexpected(rows.error());
    if (*rows != verified.size() || verified.empty()) {
      return Error(
          std::format("{}'s verify returned {} rows for {} tokens", name_, *rows, verified.size()));
    }
    const std::uint32_t pos = runner_->slots().positions(slot);
    std::vector<std::vector<float>>& kept_rows = verified_rows_.at(slot);
    kept_rows.clear();
    std::optional<chat::ToolSampling> tentative;
    if ((*branch)->tool_sampling_) tentative = *(*branch)->tool_sampling_;
    std::uint32_t m = 0;
    std::int32_t next = -1;
    for (;; ++m) {
      auto row = LogitsRow(outputs, m);
      if (!row) return std::unexpected(row.error());
      const std::uint64_t position = std::uint64_t{pos} + m + 1;
      std::int32_t token = -1;
      if (m + 1 < verified.size()) {
        auto keep = Keep(**branch, *row, verified[m + 1], position, token,
                         tentative ? &*tentative : nullptr);
        if (!keep) return std::unexpected(keep.error());
        if (*keep) {
          if (tentative) {
            auto plan = tentative->Plan(verified[m + 1]);
            if (!plan) return Error(plan.error());
            if (plan->tokens.size() != 1 || plan->tokens.front() != verified[m + 1]) {
              token = verified[m + 1];
              if (want_rows_.at(slot)) kept_rows.push_back(std::move(*row));
              next = token;
              break;
            }
            if (auto accepted = tentative->Accept(verified[m + 1]); !accepted)
              return Error(accepted.error());
          }
          // Stop at the first accepted natural terminator. Later verified rows
          // are paid native work, but do not extend this generation's lexical
          // cursor or get sampled after its already accepted end.
          const auto* session = (*branch)->generation_session_;
          token = session && session->IsStop(verified[m + 1], tentative ? &*tentative : nullptr)
                      ? verified[m + 1]
                      : -1;
        }
      } else {
        auto chosen = Choose(**branch, *row, position, tentative ? &*tentative : nullptr);
        if (!chosen) return std::unexpected(chosen.error());
        token = *chosen;
      }
      if (want_rows_.at(slot)) kept_rows.push_back(std::move(*row));
      if (token >= 0) {
        next = token;
        break;
      }
    }
    en::Verdict verdict{.keep = m + 1, .tokens = {}};
    verdict.tokens.assign(verified.begin() + 1, verified.begin() + 1 + m);
    verdict.tokens.push_back(next);
    return verdict;
  };
  if (branch_depth_ && !drafter_specs_.empty()) {
    policy.depth = [this](std::uint32_t slot, std::size_t) {
      const auto b = branch(slot);
      if (!b) return 0U;
      const auto maximum = drafter_specs_.front().max_depth;
      return sampling(**b) || maximum < 3 ? std::min(2U, maximum) : BranchDecoding(**b).Choose();
    };
    policy.observe = [this](std::uint32_t slot, std::size_t, std::uint32_t depth,
                            std::uint32_t kept, bool complete) {
      const auto b = branch(slot);
      if (b && !sampling(**b)) BranchDecoding(**b).Observe(depth, kept, complete);
    };
  }
  return policy;
}

void Llm::ResetDrafters(std::uint32_t slot) {
  if (slot < reset_.size()) reset_[slot] = true;
}

void Llm::SyncDrafters(std::span<const en::DecodeWork> work) {
  if (!speculative_) return;
  for (const en::DecodeWork& w : work) {
    if (w.slot < reset_.size() && reset_[w.slot]) {
      speculative_->Reset(w.slot);
      reset_[w.slot] = false;
    }
  }
}

void Llm::RunPlain(std::span<PreparedGeneration> units) {
  std::vector<en::DecodeWork> work;
  std::vector<bool> greedy;
  work.reserve(units.size());
  // The accepted policy uses homogeneous outputs. The internal measured
  // candidate preserves each owner's requested device-token or row result.
  const bool device =
      device_greedy_ && std::ranges::all_of(units, [](const PreparedGeneration& unit) {
        return GreedyWithoutRows(unit);
      });
  for (PreparedGeneration& unit : units) {
    unit.chosen.reset();
    const bool owner_device =
        caps_.mixed_greedy ? device_greedy_ && GreedyWithoutRows(unit) : device;
    greedy.push_back(owner_device);
    work.push_back({.slot = Slot(*unit.branch),
                    .history = unit.step.all,
                    .left = std::max<std::uint32_t>(unit.step.left, 1),
                    .greedy = owner_device,
                    .score = {},
                    .certain = unit.step.certain,
                    .certain_scores = unit.step.need_logits});
  }
  const en::DecodeReport report = plain_->Step(work);
  for (std::size_t i = 0; i < units.size(); ++i) {
    PreparedGeneration& unit = units[i];
    const en::DecodeOutcome& out = report.outcomes[i];
    if (out.refusal) {
      unit.result = Error(out.refusal->what);
      unit.failed_prefix_valid = StateUsableFor(*unit.branch);
      continue;
    }
    if (greedy[i]) {
      std::int32_t token = 0;
      if (out.outputs.size() != sizeof(token)) {
        unit.result = Error(std::format("{}'s greedy step returned no token", name_));
        unit.anchor_processed = true;
        continue;
      }
      std::memcpy(&token, out.outputs.data(), sizeof(token));
      if (token < 0 || std::cmp_greater_equal(token, logit_width_)) {
        unit.result = Error(std::format("{}'s greedy step chose token {}", name_, token));
        unit.anchor_processed = true;
        continue;
      }
      unit.chosen = token;
      ++counts_.greedy_tokens;
    } else {
      if (unit.step.forced && unit.step.need_logits) {
        for (std::size_t r = 0; r < unit.step.certain.size() + 1; ++r) {
          auto scored = LogitsRow(out.outputs, r);
          if (!scored) {
            unit.result = Error(scored.error());
            unit.anchor_processed = true;
            break;
          }
          unit.logits.push_back(std::move(*scored));
        }
        if (!unit.result) continue;
      }
      auto row = LogitsRow(
          out.outputs, unit.step.forced && unit.step.need_logits ? unit.step.certain.size() : 0);
      if (!row) {
        unit.result = Error(row.error());
        unit.anchor_processed = true;
        continue;
      }
      unit.row = std::move(*row);
    }
    unit.result = {};
  }
  if (units.size() > 1) {
    ++counts_.joined_waves;
    counts_.joined_units += units.size();
  }
}

void Llm::RunSpeculative(std::span<PreparedGeneration> units) {
  std::vector<en::DecodeWork> work;
  work.reserve(units.size());
  // The scalar SpecStepFor bridge has no session pointer. Its owned branch
  // still supplies sampling state, and need_logits supplies output intent.
  const auto device_for = [this](const PreparedGeneration& unit) {
    return device_verify_ && unit.branch != nullptr && &unit.branch->model() == this &&
           !unit.step.need_logits && !sampling(*unit.branch) &&
           unit.branch->tool_sampling_ == nullptr;
  };
  const bool device = std::ranges::all_of(units, device_for);
  for (PreparedGeneration& unit : units) {
    const std::uint32_t slot = Slot(*unit.branch);
    const bool owner_device = caps_.mixed_greedy ? device_for(unit) : device;
    verify_tokens_.at(slot) = owner_device;
    want_rows_.at(slot) = unit.step.need_logits;
    verified_rows_.at(slot).clear();
    work.push_back({.slot = slot,
                    .history = unit.step.all,
                    .left = std::max<std::uint32_t>(unit.step.left, 1),
                    .greedy = owner_device,
                    .score = {}});
  }
  SyncDrafters(work);
  const en::DecodeReport report = speculative_->Step(work);
  std::uint32_t complete = 0, committed = 0;
  for (std::size_t i = 0; i < units.size(); ++i) {
    PreparedGeneration& unit = units[i];
    const en::DecodeOutcome& out = report.outcomes[i];
    const std::uint32_t slot = work[i].slot;
    unit.drafted += out.proposed.size();
    counts_.drafted += out.proposed.size();
    ++counts_.speculative_units;
    if (out.refusal) {
      counts_.discarded += out.discarded ? 1 : 0;
      unit.result = Error(out.refusal->what);
      // Nothing was kept: the verify did not run or was undone whole.
      unit.failed_prefix_valid = StateUsableFor(*unit.branch);
      verified_rows_.at(slot).clear();
      continue;
    }
    if (wave_mode_ && unit.step.left >= verify_rows_ && out.proposed.size() + 1 == verify_rows_ &&
        out.joined.size() > out.outputs.size()) {
      ++complete;
      committed += out.kept;
    }
    counts_.accepted += out.kept - 1;
    unit.kept = out.tokens;
    if (unit.step.need_logits) {
      unit.logits = std::move(verified_rows_.at(slot));
      // A whole-block verify (DSpark) may judge more rows than the request
      // has tokens left; the orchestrator keeps only `left` of them, and
      // the rows beyond are not the request's (a row a kept token).
      if (unit.logits.size() > unit.kept.size()) unit.logits.resize(unit.kept.size());
    }
    verified_rows_.at(slot).clear();
    unit.result = {};
  }
  if (wave_mode_ &&
      std::ranges::none_of(units, [this](const auto& unit) { return sampling(*unit.branch); }))
    wave_mode_->Observe(static_cast<std::uint32_t>(units.size()),
                        execution::AdaptiveWaveMode::Mode::kSpeculative, committed, complete);
  if (units.size() > 1) {
    ++counts_.joined_waves;
    counts_.joined_units += units.size();
  }
}

Status Llm::RunPreparedGenerationWave(std::span<PreparedGeneration> prepared) {
  if (auto built = Built(); !built) return built;
  if (prepared.empty() || prepared.size() > slots_ || prepared.size() > kMaxBranches) {
    return Error(std::format("{}'s generation wave exceeds its request slots", name_));
  }
  const bool speculative = prepared.front().step.speculative;
  for (const PreparedGeneration& unit : prepared) {
    if (unit.branch == nullptr || &unit.branch->model() != this ||
        unit.step.speculative != speculative ||
        unit.step.all.size() != std::size_t{unit.step.position} + 1) {
      return Error(
          std::format("{}'s generation wave needs its own one-anchor units of one kind", name_));
    }
  }
  if (std::ranges::any_of(prepared, [](const auto& unit) { return unit.step.forced; })) {
    std::array<PreparedGeneration, kMaxBranches> known{}, ordinary{};
    std::array<std::size_t, kMaxBranches> ki{}, oi{};
    std::size_t nk = 0, no = 0;
    for (std::size_t i = 0; i < prepared.size(); ++i) {
      if (!prepared[i].step.forced) {
        oi[no] = i;
        ordinary[no++] = std::move(prepared[i]);
      } else {
        ki[nk] = i;
        known[nk++] = std::move(prepared[i]);
      }
    }
    RunPlain(std::span(known).first(nk));
    Status rest;
    if (no != 0) rest = RunPreparedGenerationWave(std::span(ordinary).first(no));
    for (std::size_t i = 0; i < nk; ++i) prepared[ki[i]] = std::move(known[i]);
    for (std::size_t i = 0; i < no; ++i) prepared[oi[i]] = std::move(ordinary[i]);
    return rest;
  }
  if (speculative) {
    if (!speculative_) return Error(std::format("{} has no drafter to speculate with", name_));
    const auto width = static_cast<std::uint32_t>(prepared.size());
    const bool sampled =
        std::ranges::any_of(prepared, [this](const auto& unit) { return sampling(*unit.branch); });
    if ((wave_mode_ &&
         wave_mode_->Choose(width, sampled) == execution::AdaptiveWaveMode::Mode::kPlain) ||
        (prepared.size() == 1 && prepared[0].step.left == 1)) {
      // Sessions still consume a speculative result, but this selected form
      // commits one target row. RunPlain retains the ordinary companion/tap
      // writes; no draft state is substituted or copied from another owner.
      RunPlain(prepared);
      for (auto& unit : prepared) {
        if (!unit.result) continue;
        auto token = unit.chosen
                         ? std::expected<std::int32_t, std::string>(*unit.chosen)
                         : Choose(*unit.branch, unit.row, std::uint64_t{unit.step.position} + 1);
        if (!token) {
          unit.result = Error(token.error());
          unit.anchor_processed = true;
          continue;
        }
        unit.kept = {*token};
        if (unit.step.need_logits) unit.logits.push_back(std::move(unit.row));
      }
      if (wave_mode_ && !sampled &&
          std::ranges::all_of(prepared, [](const auto& unit) { return unit.result.has_value(); }))
        wave_mode_->Observe(width, execution::AdaptiveWaveMode::Mode::kPlain);
    } else {
      RunSpeculative(prepared);
    }
  } else {
    RunPlain(prepared);
  }
  return GenerationCohortUsable() ? Status{}
                                  : Error(std::format("{}'s request cohort faulted", name_));
}

Status Llm::SpecStepFor(Branch& branch, std::span<const std::int32_t> all, std::uint32_t pos,
                        std::uint32_t left, std::vector<std::int32_t>& kept,
                        std::vector<std::vector<float>>* logits, std::uint64_t& drafted,
                        bool* prefix_kept) {
  if (auto built = Built(); !built) return built;
  if (!speculative_) return Error(std::format("{} has no drafter to speculate with", name_));
  PreparedGeneration unit{.session = nullptr,
                          .branch = &branch,
                          .step = {.all = all,
                                   .position = pos,
                                   .left = left,
                                   .speculative = true,
                                   .need_logits = logits != nullptr},
                          .row = {},
                          .kept = {},
                          .logits = {},
                          .drafted = 0,
                          .result = {},
                          .failed_prefix_valid = false,
                          .anchor_processed = false,
                          .chosen = std::nullopt};
  RunSpeculative(std::span(&unit, 1));
  drafted += unit.drafted;
  if (!unit.result) {
    if (prefix_kept != nullptr) *prefix_kept = unit.failed_prefix_valid;
    return unit.result;
  }
  kept = std::move(unit.kept);
  if (logits != nullptr) *logits = std::move(unit.logits);
  return {};
}

bool Llm::GenerationCohortUsable() const {
  return static_cast<bool>(runner_) && !runner_->slots().faulted();
}

Status Llm::PrepareSamplingScratchFor(Branch& branch, std::size_t rows) {
  if (rows != logit_width_) {
    return Error(std::format("{}'s sampling needs its whole head row", name_));
  }
  // Top-k may reserve 2k: the whole bound, before the first sample.
  ReserveBranchSamplingScratch(branch, 2 * std::size_t{logit_width_});
  return {};
}

void Llm::RetireSamplingScratchFor(Branch& branch) { DropBranchSamplingScratch(branch); }

// ------------------------------------------------------------- prompts

Status Llm::RunPrefill(std::span<PreparedPrefill> prepared) {
  if (prepared.empty()) return {};
  if (auto built = Built(); !built) return built;
  constexpr std::uint64_t kAllocatorMetadata = 64;
  std::uint64_t bytes = 0, peer_bytes = 0;
  if (__builtin_mul_overflow(std::uint64_t{prepared.size()},
                             std::uint64_t{sizeof(en::PrefillUnit) + sizeof(std::size_t)},
                             &bytes) ||
      __builtin_mul_overflow(std::uint64_t{active_branches_.size()},
                             std::uint64_t{sizeof(en::PrefillPeer)}, &peer_bytes) ||
      __builtin_add_overflow(bytes, peer_bytes, &bytes) ||
      __builtin_add_overflow(bytes, 3 * kAllocatorMetadata, &bytes))
    return Error("prefill adapter metadata extent overflows");
  auto metadata_payment = node_ref_.PrepayHost(bytes);
  if (!metadata_payment) {
    for (auto& unit : prepared) {
      if (unit.branch) MarkCapacityRefused(*unit.branch, true);
      unit.capacity_refused = true;
      unit.result = Error(metadata_payment.error());
    }
    return Error(metadata_payment.error());
  }
  // Declared after its payment: all temporary capacity dies before refund.
  std::vector<en::PrefillUnit> units;
  std::vector<std::size_t> of;
  units.reserve(prepared.size());
  of.reserve(prepared.size());
  for (std::size_t i = 0; i < prepared.size(); ++i) {
    auto& unit = prepared[i];
    unit.capacity_refused = false;
    if (unit.branch) MarkCapacityRefused(*unit.branch, false);
    const bool scoring = unit.session && unit.session->scoring();
    if (unit.branch == nullptr || &unit.branch->model() != this || unit.rows == 0 ||
        unit.all.size() != std::size_t{unit.past} + unit.rows) {
      unit.result = Error(std::format("{} needs its own nonempty prompt chunks", name_));
      continue;
    }
    en::PrefillHint next{.rows = unit.next.rows,
                         .want_head = unit.next.want_head,
                         .after_rows = unit.next.after_rows,
                         .after_want_head = unit.next.after_want_head};
    if (prepared.size() == 1 && unit.session && !scoring && !unit.generation) {
      const auto prompt_rows = unit.session->prompt_rows();
      const auto current_end = unit.past + unit.rows;
      if (next.rows == 0 && current_end < prompt_rows) {
        next.rows = PrefillRows(prompt_rows - current_end, max_rows_);
        next.want_head = current_end + next.rows == prompt_rows;
      }
      const auto after = current_end + next.rows;
      if (next.rows != 0 && next.after_rows == 0 && after < prompt_rows) {
        next.after_rows = PrefillRows(prompt_rows - after, max_rows_);
        next.after_want_head = after + next.after_rows == prompt_rows;
      }
    }
    units.push_back(
        {.current = {.slot = Slot(*unit.branch),
                     .stage = 0,
                     .phase = scoring           ? pl::Phase::kScore
                              : unit.generation ? pl::Phase::kDecode
                                                : pl::Phase::kPrefill,
                     .past = unit.past,
                     .rows = unit.rows,
                     .outputs = scoring ? unit.rows : (unit.want_head ? 1U : 0U),
                     .want_head = unit.want_head,
                     .greedy = unit.chosen != nullptr,
                     .row_invariant = false,
                     .alone = false},
         .work = {.tokens = unit.all.subspan(unit.past, unit.rows),
                  .bytes = {},
                  .feature = {},
                  .history = unit.all},
         .next = next,
         .prompt_end = unit.session ? std::optional(unit.session->prompt_rows()) : std::nullopt,
         .remaining = unit.session ? unit.session->continuation_rows() : std::nullopt,
         .decode_greedy = device_greedy_ && unit.session &&
                          !unit.session->continuation_sampling().value_or(true) &&
                          unit.branch->tool_sampling_ == nullptr,
         .verify_greedy = device_verify_ && unit.session &&
                          !unit.session->continuation_sampling().value_or(true) &&
                          unit.branch->tool_sampling_ == nullptr,
         .all_outputs = scoring,
         .generation_expected = unit.session && unit.session->has_continuation(),
         .sampling = unit.session && unit.session->continuation_sampling().value_or(false)});
    of.push_back(i);
  }
  std::vector<en::PrefillPeer> peers;
  peers.reserve(active_branches_.size());
  for (const auto* branch : active_branches_) {
    if (!branch || !GenerationReadyFor(*branch) ||
        std::ranges::any_of(of, [&](auto i) { return prepared[i].branch == branch; }))
      continue;
    const auto past = CursorFor(*branch);
    if (past == 0 || past >= context_) continue;
    peers.push_back(
        {.next = {.slot = Slot(*branch),
                  .stage = 0,
                  .phase = pl::Phase::kDecode,
                  .past = past,
                  .rows = 1,
                  .outputs = 1,
                  .want_head = true,
                  .greedy =
                      device_greedy_ && !sampling(*branch) && branch->tool_sampling_ == nullptr,
                  .row_invariant = speculative_ && speculative_->mode().row_invariant,
                  .alone = false},
         .ready = true,
         .verify_greedy = device_verify_ && !sampling(*branch) && branch->tool_sampling_ == nullptr,
         .sampling = sampling(*branch),
         .remaining = RemainingGenerationRowsFor(*branch)});
  }
  const en::PrefillContinuation continuation{
      .speculate = speculate_ && verify_rows_ > 1,
      .row_invariant = speculative_ && speculative_->mode().row_invariant,
      .verify_rows = verify_rows_,
      .min_verify_rows = branch_depth_ ? 1U : verify_rows_,
      .plain = [&](std::uint32_t owners, bool sampled) {
        return !speculate_ || (wave_mode_ && wave_mode_->Choose(owners, sampled) ==
                                                 execution::AdaptiveWaveMode::Mode::kPlain);
      }};
  en::PrefillOrchestrator prefill(*runner_, *composer_, prefill_);
  const auto report = prefill.StepPrepared(units, peers, continuation);
  if (report.refusal) {
    for (const auto i : of) {
      prepared[i].result = Error(report.refusal->what);
      prepared[i].capacity_refused = report.refusal->capacity_refused;
      MarkCapacityRefused(*prepared[i].branch, prepared[i].capacity_refused);
    }
    return Error(report.refusal->what);
  }
  std::size_t completed = 0;
  for (std::size_t j = 0; j < units.size(); ++j) {
    auto& unit = prepared[of[j]];
    const auto& outcome = report.outcomes()[j];
    if (outcome.refusal) {
      unit.result = Error(outcome.refusal->what);
      unit.capacity_refused = outcome.refusal->capacity_refused;
      MarkCapacityRefused(*unit.branch, unit.capacity_refused);
      continue;
    }
    ++completed;
    unit.retired = true;
    if (!unit.generation) ResetDrafters(units[j].current.slot);
    if (unit.chosen) {
      std::int32_t token = 0;
      if (outcome.outputs.size() != sizeof(token)) {
        unit.result = Error("mixed decode returned no complete device token");
        continue;
      }
      std::memcpy(&token, outcome.outputs.data(), sizeof(token));
      if (token < 0 || std::cmp_greater_equal(token, logit_width_)) {
        unit.result = Error("mixed decode returned an invalid device token");
        continue;
      }
      *unit.chosen = token;
      ++counts_.greedy_tokens;
    } else if (unit.want_head) {
      const bool scoring = unit.session && unit.session->scoring();
      const auto expected_rows = scoring ? unit.rows : 1U;
      auto actual_rows = HeadRows(outcome.outputs);
      if (!actual_rows || *actual_rows != expected_rows) {
        unit.result = Error(actual_rows ? "scoring result row count differs from retired work"
                                        : actual_rows.error());
        continue;
      }
      if (scoring)
        unit.scored = {.bytes = outcome.outputs,
                       .first = unit.past,
                       .rows = unit.rows,
                       .width = static_cast<std::uint32_t>(logit_width_)};
      auto row = LogitsRow(outcome.outputs, expected_rows - 1);
      if (!row) {
        unit.result = Error(row.error());
        continue;
      }
      if (unit.logits) *unit.logits = std::move(*row);
    } else if (unit.logits) {
      unit.logits->clear();
    }
    unit.result = {};
  }
  if (completed > 1 && report.steps < completed) {
    const bool mixed =
        std::ranges::any_of(prepared, [](const auto& unit) { return unit.generation != nullptr; });
    auto& steps = mixed ? counts_.joined_causal_steps : counts_.joined_prefill_steps;
    auto& rows = mixed ? counts_.joined_causal_rows : counts_.joined_prefill_rows;
    ++steps;
    for (std::size_t j = 0; j < units.size(); ++j)
      if (!report.outcomes()[j].refusal) rows += units[j].current.rows;
  }
  return GenerationCohortUsable() ? Status{}
                                  : Error(std::format("{}'s request cohort faulted", name_));
}

Status Llm::RunPreparedCausalWave(std::span<PreparedPrefill> prompts,
                                  std::span<PreparedGeneration> generations) {
  if (!supports_causal_waves() || prompts.size() + generations.size() > caps_.max_segments ||
      prompts.size() + generations.size() > kMaxBranches)
    return Error("mixed causal dispatch exceeds its actual mechanical policy");
  if (std::ranges::any_of(generations, [](const auto& unit) { return unit.step.forced; })) {
    // Certain causal spans have their own semantic rows (and possibly genuine
    // all-row scores). Keep those owners together in the common decoder;
    // ordinary mixed prompt/decode owners retain their existing joined wave.
    std::array<PreparedGeneration, kMaxBranches> certain{}, ordinary{};
    std::array<std::size_t, kMaxBranches> ci{}, oi{};
    std::size_t nc = 0, no = 0;
    for (std::size_t i = 0; i < generations.size(); ++i) {
      if (generations[i].step.forced) {
        ci[nc] = i;
        certain[nc++] = std::move(generations[i]);
      } else {
        oi[no] = i;
        ordinary[no++] = std::move(generations[i]);
      }
    }
    auto result = RunPreparedGenerationWave(std::span(certain).first(nc));
    if (result && (!prompts.empty() || no != 0))
      result = RunPreparedCausalWave(prompts, std::span(ordinary).first(no));
    for (std::size_t i = 0; i < nc; ++i) generations[ci[i]] = std::move(certain[i]);
    for (std::size_t i = 0; i < no; ++i) generations[oi[i]] = std::move(ordinary[i]);
    return result;
  }
  std::array<PreparedPrefill, kMaxBranches> storage{};
  std::array<std::size_t, kMaxBranches> source{};
  std::size_t count = 0;
  for (std::size_t i = 0; i < prompts.size(); ++i) {
    storage[count] = std::move(prompts[i]);
    source[count++] = i;
  }
  for (std::size_t i = 0; i < generations.size(); ++i) {
    auto& unit = generations[i];
    if (unit.step.speculative) return Error("mixed causal dispatch cannot replace verification");
    unit.chosen.reset();
    storage[count] = {.session = nullptr,
                      .branch = unit.branch,
                      .all = unit.step.all,
                      .past = unit.step.position,
                      .rows = 1,
                      .want_head = true,
                      .next = {},
                      .logits = &unit.row,
                      .result = {},
                      .generation = unit.session,
                      .chosen = device_greedy_ && GreedyWithoutRows(unit) ? &unit.chosen : nullptr};
    source[count++] = prompts.size() + i;
  }
  // Both maps are caller-bounded by the existing slot representation; all
  // dynamic native dispatch metadata is paid by RunPrefill before construction.
  auto prepared = std::span(storage).first(count);
  // Keep the source map in the same order as each moved descriptor.
  for (std::size_t i = 1; i < count; ++i)
    for (std::size_t j = i; j && Slot(*prepared[j].branch) < Slot(*prepared[j - 1].branch); --j) {
      std::swap(prepared[j], prepared[j - 1]);
      std::swap(source[j], source[j - 1]);
    }
  const auto result = RunPrefill(prepared);
  for (std::size_t i = 0; i < count; ++i) {
    if (source[i] < prompts.size())
      prompts[source[i]] = std::move(prepared[i]);
    else {
      auto& unit = generations[source[i] - prompts.size()];
      unit.result = std::move(prepared[i].result);
      unit.anchor_processed = !unit.result && prepared[i].retired;
      if (!unit.result) unit.failed_prefix_valid = StateUsableFor(*unit.branch);
    }
  }
  return result;
}

std::uint32_t Llm::ScoreChunkRowsFor(const Branch& /*branch*/) const {
  if (!runner_) return 0;
  return std::min(max_rows_, runner_->slots().layout(0).max_rows);
}

Status Llm::RunPreparedPrefillWave(std::span<PreparedPrefill> prepared) {
  if (prepared.empty() || prepared.size() > prefill_wave_capacity()) {
    return Error(std::format("{}'s prompt wave exceeds its capability", name_));
  }
  if (prepared.size() != 1 || !prepared[0].session || !prepared[0].session->scoring())
    return RunPrefill(prepared);
  auto& unit = prepared[0];
  const auto original = unit.all;
  // This only chooses a candidate; exact selected target/companion resources
  // and all results still go through complete-batch admission before progress.
  // A head row needs device output, pinned DMA and immutable host publication.
  const auto free = node_ref_.FreeBytes();
  if (!free) {
    unit.result = Error(free.error());
    return unit.result;
  }
  std::uint64_t row_cost = 0;
  if (__builtin_mul_overflow(std::uint64_t{logit_width_}, std::uint64_t{3 * sizeof(float)},
                             &row_cost) ||
      row_cost == 0) {
    unit.result = Error("scoring head resource extent overflows");
    return unit.result;
  }
  const auto candidate = std::max<std::uint64_t>(1, *free / row_cost);
  unit.rows = static_cast<std::uint32_t>(std::min(std::uint64_t{unit.rows}, candidate));
  for (;;) {
    unit.all = original.first(std::size_t{unit.past} + unit.rows);
    unit.next = {};
    unit.result = {};
    unit.scored = {};
    const auto result = RunPrefill(prepared);
    if (!unit.capacity_refused || unit.rows == 1) return result;
    // A clean capacity refusal happened before every numeric batch child.
    // Retry the complete schedule, preserving the exact token/history prefix.
    unit.rows = std::max(1U, unit.rows / 2);
  }
}

Status Llm::RunPrefillChunkFor(Branch& branch, std::span<const std::int32_t> all,
                               std::uint32_t past, bool want_head, std::vector<float>& logits,
                               PrefillHint next) {
  if (past >= all.size()) return Error(std::format("{} needs a nonempty prompt chunk", name_));
  PreparedPrefill unit{.session = nullptr,
                       .branch = &branch,
                       .all = all,
                       .past = past,
                       .rows = static_cast<std::uint32_t>(all.size() - past),
                       .want_head = want_head,
                       .next = next,
                       .logits = &logits,
                       .result = {}};
  if (auto ran = RunPrefill(std::span(&unit, 1)); !ran && unit.result) return ran;
  return unit.result;
}

Status Llm::PreparePrefillStateFor(Branch& branch, std::uint32_t past, std::uint32_t rows) {
  if (auto built = Built(); !built) return built;
  const std::uint32_t slot = Slot(branch);
  if (past != runner_->slots().positions(slot) || rows == 0 || past > context_ ||
      rows > context_ - past) {
    return Error(std::format("{}'s prefill funding needs an owned bounded continuation", name_));
  }
  return runner_->slots().ReserveStateThrough(slot, past + rows);
}

Status Llm::RunChunkFor(Branch& branch, std::span<const std::int32_t> all, std::uint32_t past,
                        std::vector<float>& logits) {
  if (auto built = Built(); !built) return built;
  if (past >= all.size()) return Error(std::format("{} needs a nonempty chunk", name_));
  // One row after the state: a decode row (a generation's, a scorer's).
  if (all.size() == std::size_t{past} + 1) {
    PreparedGeneration unit{.session = nullptr,
                            .branch = &branch,
                            .step = {.all = all,
                                     .position = past,
                                     .left = 1,
                                     .speculative = false,
                                     .need_logits = true},
                            .row = {},
                            .kept = {},
                            .logits = {},
                            .drafted = 0,
                            .result = {},
                            .failed_prefix_valid = false,
                            .anchor_processed = false,
                            .chosen = std::nullopt};
    RunPlain(std::span(&unit, 1));
    if (!unit.result) return unit.result;
    logits = std::move(unit.row);
    return {};
  }
  // A longer chunk: prompt steps of the model's chunk, the last headed.
  for (std::uint32_t at = past; at < all.size();) {
    const auto rows = static_cast<std::uint32_t>(std::min<std::size_t>(max_rows_, all.size() - at));
    const bool last = std::size_t{at} + rows == all.size();
    PreparedPrefill unit{.session = nullptr,
                         .branch = &branch,
                         .all = all.first(std::size_t{at} + rows),
                         .past = at,
                         .rows = rows,
                         .want_head = last,
                         .next = {},
                         .logits = &logits,
                         .result = {}};
    if (auto ran = RunPrefill(std::span(&unit, 1)); !ran && unit.result) return ran;
    if (!unit.result) return unit.result;
    at += rows;
  }
  return {};
}

// ------------------------------------------------------------- the state

Status Llm::SettleFor(Branch& branch) {
  if (auto built = Built(); !built) return built;
  const std::uint32_t slot = Slot(branch);
  en::SlotLifecycle& slots = runner_->slots();
  if (!slots.spilled(slot) && slots.owed(slot)) {
    if (auto settled = slots.Settle(slot); !settled) return settled;
  }
  return slots.usable(slot) ? Status{}
                            : Error(std::format("{}'s conversation state is unusable", name_));
}

Status Llm::ClearStateFor(Branch& branch) {
  if (auto built = Built(); !built) return built;
  if (branch_depth_)
    BranchDecoding(branch) = execution::AdaptiveDepth(branch_depth_, depth_cost_ratio_);
  return runner_->slots().Clear(Slot(branch));
}

bool Llm::StateUsableFor(const Branch& branch) const {
  // Prefix reuse includes healthy kept state that still needs hydration.
  // Execution guards continue to reject its pending progress replacement.
  return runner_ &&
         (runner_->slots().usable(Slot(branch)) || runner_->slots().restorable(Slot(branch)));
}

bool Llm::StateRefusedFor(const Branch& branch) const {
  return branch.capacity_refused() || (runner_ && runner_->slots().state_refused(Slot(branch)));
}

Status Llm::ReleaseIdleStateFor(Branch& branch) {
  if (auto built = Built(); !built) return built;
  if (branch_depth_)
    BranchDecoding(branch) = execution::AdaptiveDepth(branch_depth_, depth_cost_ratio_);
  return runner_->slots().ClearIdle(Slot(branch));
}

Status Llm::SpillFor(Branch& branch) {
  if (auto built = Built(); !built) return built;
  return runner_->slots().Spill(Slot(branch));
}

Status Llm::RestoreFor(Branch& branch) {
  if (auto built = Built(); !built) return built;
  return runner_->slots().Restore(Slot(branch));
}

bool Llm::SpilledFor(const Branch& branch) const {
  return runner_ && runner_->slots().spilled(Slot(branch));
}

std::uint64_t Llm::SpilledBytesFor(const Branch& branch) const {
  return runner_ ? runner_->slots().spilled_bytes(Slot(branch)) : 0;
}

std::uint64_t Llm::RetainedSpillBytesFor(const Branch& branch) const {
  return runner_ ? runner_->slots().retained_spill_bytes(Slot(branch)) : 0;
}

Status Llm::ReleaseRetainedSpillFor(Branch& branch) {
  if (auto built = Built(); !built) return built;
  return runner_->slots().ReleaseSpillFiles(Slot(branch));
}

bool Llm::LeasedFor(const Branch& branch) const {
  return runner_ && runner_->slots().held(Slot(branch));
}

std::uint64_t Llm::RefusedBytesFor(const Branch& branch) const {
  if (!runner_) return 0;
  const en::SlotLifecycle& slots = runner_->slots();
  const std::uint32_t slot = Slot(branch);
  // A refused restore asks for what the slot's files hold.
  return slots.spilled(slot) ? slots.spilled_bytes(slot) : slots.refused_bytes(slot);
}

std::uint64_t Llm::SpillWriteBytesFor(const Branch& branch) const {
  return runner_ ? runner_->slots().spill_write_bytes(Slot(branch)) : 0;
}

Status Llm::PrepareDecodeStateFor(Branch& branch, std::uint32_t pos, std::uint32_t left,
                                  std::uint32_t certain_rows) {
  if (auto built = Built(); !built) return built;
  const std::uint32_t slot = Slot(branch);
  if (pos != runner_->slots().positions(slot) || pos >= context_) {
    return Error(std::format("{}'s decode position differs from its completed state", name_));
  }
  // A verify's rows (the anchor and every proposal) at most, within the
  // context: the decoding orchestrator never takes more.
  const std::uint32_t rows =
      certain_rows != 0 ? certain_rows + 1
      : speculate_      ? std::max<std::uint32_t>(std::min({left, verify_rows_, context_ - pos}), 1)
                        : 1;
  return runner_->slots().ReserveStateThrough(slot, pos + rows);
}

std::uint64_t Llm::UsedStateBytesFor(const Branch& branch) const {
  return runner_ ? runner_->slots().used_bytes(Slot(branch)) : 0;
}

std::vector<en::LiveState::Range> Llm::UsedStateRangesFor(const Branch& branch) const {
  std::vector<en::LiveState::Range> out;
  if (!runner_) return out;
  const en::SlotLifecycle& slots = runner_->slots();
  for (std::size_t s = 0; s < slots.stages(); ++s) {
    for (const en::LiveState::Range& r : slots.live(Slot(branch), s).used_ranges()) {
      out.push_back({.region = s, .offset = r.offset, .bytes = r.bytes});
    }
  }
  return out;
}

Status Llm::SaveUsedStateFor(Branch& branch, void* host,
                             std::span<const en::LiveState::Range> ranges) {
  if (auto built = Built(); !built) return built;
  return runner_->slots().CopyState(Slot(branch), host, ranges, /*to_host=*/true);
}

std::expected<std::vector<en::LiveState::Range>, std::string> Llm::CheckpointRangesFor(
    const Branch& branch, std::uint32_t positions) const {
  if (auto built = Built(); !built) return std::unexpected(built.error());
  const std::uint32_t slot = Slot(branch);
  if (positions != runner_->slots().positions(slot)) {
    return Error(std::format("{}'s checkpoint boundary differs from its completed state", name_));
  }
  return runner_->slots().CheckpointRanges(slot, positions);
}

Status Llm::CopyCheckpointStateFor(Branch& branch, void* host,
                                   std::span<const en::LiveState::Range> ranges, bool to_host) {
  if (auto built = Built(); !built) return built;
  return runner_->slots().CopyState(Slot(branch), host, ranges, to_host);
}

Status Llm::CheckCheckpointMetadataFor(const Branch& /*branch*/, std::uint32_t position,
                                       std::uint32_t cursor,
                                       std::span<const en::LiveState::Range> footprint) const {
  if (auto built = Built(); !built) return built;
  if (position != cursor) {
    return Error(std::format("{}'s checkpoint cursor differs from its positions", name_));
  }
  return runner_->slots().ValidateFootprint(position, footprint);
}

Status Llm::PreparePositionRestoreFor(Branch& branch, std::uint32_t position, std::uint32_t cursor,
                                      std::span<const en::LiveState::Range> footprint,
                                      std::string_view layout) {
  if (auto checked = CheckCheckpointMetadataFor(branch, position, cursor, footprint); !checked) {
    return checked;
  }
  return runner_->slots().PrepareRestore(Slot(branch), position, footprint, layout);
}

Status Llm::CompletePositionRestoreFor(Branch& branch, std::uint32_t position) {
  if (auto built = Built(); !built) return built;
  if (auto completed = runner_->slots().CompleteRestore(Slot(branch), position); !completed) {
    return completed;
  }
  ResetDrafters(Slot(branch));
  return {};
}

Status Llm::RestoreSnapshotFor(Branch& branch, void* host,
                               std::span<const en::LiveState::Range> ranges, std::uint32_t cursor,
                               std::string_view layout) {
  // The authenticated capture names its original layout.
  if (auto prepared = PreparePositionRestoreFor(branch, cursor, cursor, ranges, layout);
      !prepared) {
    return prepared;
  }
  if (auto copied = CopyCheckpointStateFor(branch, host, ranges, /*to_host=*/false); !copied) {
    return copied;
  }
  return CompletePositionRestoreFor(branch, cursor);
}

Status Llm::AdoptPositionFor(Branch& branch, std::span<const std::int32_t> tokens,
                             std::uint32_t cursor, std::span<const en::LiveState::Range> ranges,
                             std::string_view layout) {
  if (auto built = Built(); !built) return built;
  if (cursor == 0 || cursor != tokens.size()) {
    return Error(std::format("{}'s kept positions differ from its token history", name_));
  }
  if (auto adopted = runner_->slots().Adopt(Slot(branch), cursor, ranges, layout); !adopted) {
    return adopted;
  }
  ResetDrafters(Slot(branch));
  return {};
}

std::uint32_t Llm::CursorFor(const Branch& branch) const {
  return runner_ ? runner_->slots().positions(Slot(branch)) : 0;
}

}  // namespace llmp::runtime
