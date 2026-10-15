// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/engine_llm.h"

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

EngineLlm::EngineLlm(en::PagedNode& node, const config::ModelEntry& entry,
                     const ModelSettings& settings, const config::RuntimeRoles& roles, int index,
                     EngineBuild build)
    : node_ref_(node),
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

EngineLlm::~EngineLlm() = default;

Status EngineLlm::Built() const {
  if (!runner_ || !composer_ || !plain_) return Error(name_ + " is not set up");
  if (auto identity = ExecutionIdentity(); !identity) return Error(identity.error());
  return {};
}

// ------------------------------------------------------------------ its life

en::PagedModel& EngineLlm::paged() {
  base::Check(runner_ != nullptr, "an engine-backed model's runner before its Setup");
  return *runner_;
}

Status EngineLlm::Setup() {
  if (runner_ != nullptr) return Error(std::format("model {}: set up already", name_));
  if (!build_) return Error(std::format("model {}: no component pipeline builds it", name_));
  auto parts = build_(settings_);
  if (!parts) return Error(std::format("model {}: {}", name_, parts.error()));
  if (parts->logit_width == 0) return Error(std::format("model {}: its head has no logits", name_));
  // Its tokenizer and chat template, from its artifact (D-067); a model
  // without one (the tests') has no chat turns.
  if (entry_.artifact) {
    auto artifact = OpenInstalled(store_, *entry_.artifact);
    if (!artifact) return Error(artifact.error());
    if (auto used = UseChatAssets(*artifact, entry_, parts->startup_cap); !used) return used;
  }
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
  slots.invalidate = [this, also = std::move(slots.invalidate)](std::uint32_t slot) {
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
  auto runner =
      en::ModelRunner::Create(node_ref_, std::move(parts->config), std::move(parts->source), index_,
                              static_cast<std::uint32_t>(index_));
  if (!runner) return Error(std::format("model {}: {}", name_, runner.error()));
  runner_ = std::move(*runner);
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

Status EngineLlm::Register() {
  if (auto built = Built(); !built) return built;
  return runner_->Register();
}

Status EngineLlm::Bind() {
  if (auto built = Built(); !built) return built;
  return runner_->Bind();
}

std::uint64_t EngineLlm::activations_needed() const {
  return runner_ ? runner_->activations_needed() : 0;
}
std::uint64_t EngineLlm::pool_needed() const { return runner_ ? runner_->pool_needed() : 0; }
std::uint64_t EngineLlm::host_input_bytes() const {
  // Beside the runner's staging: per slot a retained frontier and a
  // prepared row, and the sampling candidates' bound (2 × vocabulary).
  const std::uint64_t heap = std::uint64_t{slots_} * logit_width_ *
                             (2 * sizeof(float) + 2 * sizeof(execution::SamplingCandidate));
  return (runner_ ? runner_->host_input_bytes() : 0) + heap;
}
std::uint64_t EngineLlm::plan_floor_bytes() const {
  return runner_ ? runner_->plan_floor_bytes() : 0;
}

std::string EngineLlm::slots_report() const {
  return std::format("{} request slot{} ({}: {})", slots_, slots_ == 1 ? "" : "s",
                     SourceName(settings_.max_slots.source), settings_.max_slots.basis);
}

void EngineLlm::ReclaimCandidates(std::uint32_t owner, bool running,
                                  std::vector<memory::ReclaimCandidate>& out) {
  if (runner_) runner_->ReclaimCandidates(owner, running, out);
}
std::uint64_t EngineLlm::Reclaim(memory::ReclaimKind kind, std::uint64_t id) {
  return runner_ ? runner_->Reclaim(kind, id) : 0;
}
std::vector<catalog::ExtentId> EngineLlm::weights() const {
  return runner_ ? runner_->weights() : std::vector<catalog::ExtentId>{};
}
std::vector<catalog::ExtentId> EngineLlm::state() const {
  return runner_ ? runner_->slots().state() : std::vector<catalog::ExtentId>{};
}
std::vector<catalog::ExtentId> EngineLlm::kept_state() const {
  return runner_ ? runner_->slots().kept_state() : std::vector<catalog::ExtentId>{};
}
std::vector<catalog::ExtentId> EngineLlm::unchanged_state() const {
  return runner_ ? runner_->slots().unchanged_state() : std::vector<catalog::ExtentId>{};
}
void EngineLlm::StateWrittenBack(bool whole) {
  if (runner_) runner_->slots().StateWrittenBack(whole);
}
const catalog::Closure& EngineLlm::everything() const {
  base::Check(runner_ != nullptr, "an engine-backed model's closure before its Setup");
  return runner_->slots().everything();
}
const catalog::Closure& EngineLlm::request_closure() const {
  base::Check(runner_ != nullptr, "an engine-backed model's closure before its Setup");
  return runner_->slots().execution();
}
Status EngineLlm::PrepareDefaultRequest() {
  std::array<Branch*, 1> selected{&default_branch()};
  return SelectBranches(selected);
}
std::uint64_t EngineLlm::weight_read_bytes() const {
  return runner_ ? runner_->weight_read_bytes() : 0;
}
Status EngineLlm::CheckPlaces() {
  if (auto built = Built(); !built) return built;
  return runner_->slots().CheckPlaces();
}
void EngineLlm::DropPlans() {
  if (runner_) runner_->DropPlans();
}

Status EngineLlm::RecoverInPlace() {
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

GraphCounts EngineLlm::graphs() const {
  if (!runner_) return {};
  const en::GraphStats& g = runner_->graph_stats();
  return {.eager = g.eager,
          .captured = g.captured,
          .replayed = g.replayed,
          .refused = g.refused,
          .kept = runner_->plans().graphs()};
}

std::string EngineLlm::violations() const {
  return runner_ ? runner_->coverage().first_violation : std::string();
}

std::string EngineLlm::extra() const {
  return std::format(
      R"({{"engine":"component","architecture":"{}","slots":{},"speculative":{},"verify_rows":{},"joined_prefill_steps":{},"joined_prefill_rows":{},"joined_waves":{},"joined_units":{},"speculative_units":{},"drafted":{},"accepted":{},"discarded":{},"greedy_tokens":{},"runner":{}}})",
      architecture_, slots_, speculate_, verify_rows_, counts_.joined_prefill_steps,
      counts_.joined_prefill_rows, counts_.joined_waves, counts_.joined_units,
      counts_.speculative_units, counts_.drafted, counts_.accepted, counts_.discarded,
      counts_.greedy_tokens, runner_ ? runner_->report_json() : std::string("{}"));
}

// ------------------------------------------------------------------ the LLM

void EngineLlm::Defaults(chat::Conversation& conversation) const {
  if (conversation.enable_thinking) return;
  if (!chat_.thinking) {
    conversation.enable_thinking = false;
  } else if (chat_.default_thinking) {
    conversation.enable_thinking = *chat_.default_thinking;
  }
}

Status EngineLlm::CheckConversation(const chat::Conversation& conversation) const {
  if (!chat_.thinking && conversation.enable_thinking.value_or(false)) {
    return Error(std::format("{}'s thought-channel output is not served yet", name_));
  }
  if (!chat_.tools && !conversation.tools.empty()) {
    return Error(std::format("{}'s generated tool calls are not served yet", name_));
  }
  return {};
}

Status EngineLlm::SelectBranches(std::span<Branch* const> active) {
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

std::uint64_t EngineLlm::StateBytesThrough(std::uint32_t positions) const {
  if (!runner_) return 0;
  const en::SlotLifecycle& slots = runner_->slots();
  std::uint64_t total = 0;
  for (std::size_t s = 0; s < slots.stages(); ++s) {
    const model::StateLayoutDerived& layout = slots.layout(s);
    auto used =
        model::UsedThrough(layout, std::min(positions, layout.context), slots.read_align(s));
    if (!used) return 0;
    std::vector<en::LiveState::Range> ranges;
    ranges.reserve(used->size());
    for (const auto& r : *used)
      ranges.push_back({.region = 0, .offset = r.offset, .bytes = r.bytes});
    auto bytes = slots.live(0, s).UsedBytesOf(ranges);
    if (!bytes) return 0;
    total += *bytes;
  }
  return total;
}

std::size_t EngineLlm::generation_wave_capacity() const { return slots_; }

std::size_t EngineLlm::prefill_wave_capacity() const {
  return caps_.join_prefill ? std::min<std::size_t>(slots_, std::max(caps_.max_segments, 1U)) : 1;
}

bool EngineLlm::CompatiblePrefill(std::uint32_t past, std::uint32_t rows, std::uint32_t peer_past,
                                  std::uint32_t peer_rows) const {
  if (!caps_.join_prefill || !composer_ || rows < 2 || rows != peer_rows) return false;
  if (past > context_ || rows > context_ - past || peer_past > context_ ||
      peer_rows > context_ - peer_past) {
    return false;
  }
  if (!caps_.equal_prefill) return true;
  // Equal reads at the joined alignment (the composer joins no others).
  const std::uint32_t align = composer_->AlignFor(0, 2);
  const auto read = [&](std::uint64_t end) {
    return std::min<std::uint64_t>((end + align - 1) / align * align, context_);
  };
  return read(std::uint64_t{past} + rows) == read(std::uint64_t{peer_past} + peer_rows);
}

std::uint32_t EngineLlm::StepTokenBoundFor(const Branch& /*branch*/, std::uint32_t left) const {
  return std::min(left, speculate_ ? verify_rows_ : 1U);
}

// ------------------------------------------------------------- D-105

std::expected<std::optional<base::Sha256Digest>, std::string> EngineLlm::ExecutionIdentity() const {
  if (!runner_) return std::unexpected("the execution source is not set up");
  auto actual = runner_->ExecutionIdentity();
  if (!actual) return std::unexpected(actual.error());
  if (*actual != execution_identity_)
    return std::unexpected("the execution identity changed after setup");
  return execution_identity_;
}

std::string EngineLlm::KeptLayout() const {
  return runner_ ? runner_->slots().kept_layout_id() : std::string();
}

std::vector<std::uint64_t> EngineLlm::KeptRegions() const {
  std::vector<std::uint64_t> out;
  if (!runner_) return out;
  for (std::size_t s = 0; s < runner_->slots().stages(); ++s) {
    out.push_back(runner_->slots().live(0, s).mapped_bytes(0));
  }
  return out;
}

std::vector<std::uint64_t> EngineLlm::KeptLayouts() const {
  std::vector<std::uint64_t> out;
  if (!runner_) return out;
  for (std::size_t s = 0; s < runner_->slots().stages(); ++s) {
    out.push_back(runner_->slots().live(0, s).bytes(0));
  }
  return out;
}

const en::LiveState* EngineLlm::KeptLiveFor(const Branch& branch) const {
  return runner_ ? &runner_->slots().live(Slot(branch), 0) : nullptr;
}

std::vector<const en::LiveState*> EngineLlm::KeptLivesFor(const Branch& branch) const {
  std::vector<const en::LiveState*> out;
  if (!runner_) return out;
  for (std::size_t s = 0; s < runner_->slots().stages(); ++s) {
    out.push_back(&runner_->slots().live(Slot(branch), s));
  }
  return out;
}

bool EngineLlm::KeptWholeFor(const Branch& branch) const {
  return runner_ && runner_->slots().kept_whole(Slot(branch));
}

bool EngineLlm::OwedFor(const Branch& branch) const {
  return runner_ && runner_->slots().owed(Slot(branch));
}

void EngineLlm::SetStageSpillPlaces(const StageSpillPlace& place) { stage_place_ = place; }

// ------------------------------------------------------------- outputs

std::expected<std::size_t, std::string> EngineLlm::HeadRows(
    std::span<const std::byte> outputs) const {
  const std::size_t row = std::size_t{logit_width_} * sizeof(float);
  if (row == 0 || outputs.empty() || outputs.size() % row != 0) {
    return Error(std::format("{}'s step returned {} output bytes, not rows of {} logits", name_,
                             outputs.size(), logit_width_));
  }
  return outputs.size() / row;
}

std::expected<std::vector<float>, std::string> EngineLlm::LogitsRow(
    std::span<const std::byte> outputs, std::size_t row) const {
  auto rows = HeadRows(outputs);
  if (!rows) return std::unexpected(rows.error());
  if (row >= *rows) return Error(std::format("{}'s step returned no row {}", name_, row));
  std::vector<float> logits(logit_width_);
  std::memcpy(logits.data(), outputs.data() + (row * logits.size() * sizeof(float)),
              logits.size() * sizeof(float));
  return logits;
}

// ------------------------------------------------------------- decoding

en::DecodingPolicy EngineLlm::Policy() {
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
    std::uint32_t m = 0;
    std::int32_t next = -1;
    for (;; ++m) {
      auto row = LogitsRow(outputs, m);
      if (!row) return std::unexpected(row.error());
      const std::uint64_t position = std::uint64_t{pos} + m + 1;
      std::int32_t token = -1;
      if (m + 1 < verified.size()) {
        auto keep = Keep(**branch, *row, verified[m + 1], position, token);
        if (!keep) return std::unexpected(keep.error());
        if (*keep) token = -1;
      } else {
        auto chosen = Choose(**branch, *row, position);
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

void EngineLlm::ResetDrafters(std::uint32_t slot) {
  if (slot < reset_.size()) reset_[slot] = true;
}

void EngineLlm::SyncDrafters(std::span<const en::DecodeWork> work) {
  if (!speculative_) return;
  for (const en::DecodeWork& w : work) {
    if (w.slot < reset_.size() && reset_[w.slot]) {
      speculative_->Reset(w.slot);
      reset_[w.slot] = false;
    }
  }
}

void EngineLlm::RunPlain(std::span<PreparedGeneration> units) {
  std::vector<en::DecodeWork> work;
  std::vector<bool> greedy;
  work.reserve(units.size());
  // A wave chooses on the device only when every unit may (a step's greedy
  // segments are all of them): one that keeps its rows or samples keeps
  // the whole wave on rows, joined, as the M3 classes' homogeneous waves.
  const bool device =
      device_greedy_ && std::ranges::all_of(units, [](const PreparedGeneration& unit) {
        return GreedyWithoutRows(unit);
      });
  for (PreparedGeneration& unit : units) {
    unit.chosen.reset();
    greedy.push_back(device);
    work.push_back({.slot = Slot(*unit.branch),
                    .history = unit.step.all,
                    .left = std::max<std::uint32_t>(unit.step.left, 1),
                    .greedy = device,
                    .score = {}});
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
      auto row = LogitsRow(out.outputs, 0);
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

void EngineLlm::RunSpeculative(std::span<PreparedGeneration> units) {
  std::vector<en::DecodeWork> work;
  work.reserve(units.size());
  // The scalar SpecStepFor bridge has no session pointer. Its owned branch
  // still supplies sampling state, and need_logits supplies output intent.
  const bool device =
      device_verify_ && std::ranges::all_of(units, [this](const PreparedGeneration& unit) {
        return unit.branch != nullptr && &unit.branch->model() == this && !unit.step.need_logits &&
               !sampling(*unit.branch);
      });
  for (PreparedGeneration& unit : units) {
    const std::uint32_t slot = Slot(*unit.branch);
    verify_tokens_.at(slot) = device;
    want_rows_.at(slot) = unit.step.need_logits;
    verified_rows_.at(slot).clear();
    work.push_back({.slot = slot,
                    .history = unit.step.all,
                    .left = std::max<std::uint32_t>(unit.step.left, 1),
                    .greedy = device,
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

Status EngineLlm::RunPreparedGenerationWave(std::span<PreparedGeneration> prepared) {
  if (auto built = Built(); !built) return built;
  if (prepared.empty() || prepared.size() > slots_) {
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

Status EngineLlm::SpecStepFor(Branch& branch, std::span<const std::int32_t> all, std::uint32_t pos,
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

Status EngineLlm::SpecStep(std::span<const std::int32_t> all, std::uint32_t pos, std::uint32_t left,
                           std::vector<std::int32_t>& kept, std::vector<std::vector<float>>* logits,
                           std::uint64_t& drafted) {
  return SpecStepFor(default_branch(), all, pos, left, kept, logits, drafted, nullptr);
}

bool EngineLlm::GenerationCohortUsable() const {
  return runner_ != nullptr && !runner_->slots().faulted();
}

Status EngineLlm::PrepareSamplingScratchFor(Branch& branch, std::size_t rows) {
  if (rows != logit_width_) {
    return Error(std::format("{}'s sampling needs its whole head row", name_));
  }
  // Top-k may reserve 2k: the whole bound, before the first sample.
  ReserveBranchSamplingScratch(branch, 2 * std::size_t{logit_width_});
  return {};
}

void EngineLlm::RetireSamplingScratchFor(Branch& branch) { DropBranchSamplingScratch(branch); }

// ------------------------------------------------------------- prompts

Status EngineLlm::RunPrefill(std::span<PreparedPrefill> prepared) {
  if (auto built = Built(); !built) return built;
  std::vector<en::PendingWork> pending;
  std::vector<en::SegmentWork> work;
  std::vector<std::size_t> of;  // each pending work's unit
  for (std::size_t i = 0; i < prepared.size(); ++i) {
    PreparedPrefill& unit = prepared[i];
    if (unit.branch == nullptr || &unit.branch->model() != this || unit.rows == 0 ||
        unit.all.size() != std::size_t{unit.past} + unit.rows) {
      unit.result = Error(std::format("{} needs its own nonempty prompt chunks", name_));
      continue;
    }
    if (unit.next.rows > 0) {
      prefill_done_pos_.at(Slot(*unit.branch)) = 0;
    }
    pending.push_back({.slot = Slot(*unit.branch),
                       .stage = 0,
                       .phase = pl::Phase::kPrefill,
                       .past = unit.past,
                       .rows = unit.rows,
                       .outputs = unit.want_head ? 1U : 0U,
                       .want_head = unit.want_head,
                       .greedy = false,
                       .row_invariant = false,
                       .alone = false});
    work.push_back({.tokens = unit.all.subspan(unit.past, unit.rows), .bytes = {}});
    of.push_back(i);
  }
  // The same owners' next two chunks, as their sessions said (an owner
  // whose prompt ends there drops out), as the steps they will run in:
  // together when Llm would join them (CompatiblePrefill), else each alone.
  const auto future = [&](std::span<const std::size_t> owners,
                          bool after) -> std::vector<pl::StepShape> {
    std::vector<en::PendingWork> next;
    for (const std::size_t j : owners) {
      const PreparedPrefill& unit = prepared[of[j]];
      const PrefillHint& hint = unit.next;
      const std::uint32_t rows = after ? hint.after_rows : hint.rows;
      if (rows == 0 || (after && hint.rows == 0)) continue;
      const bool head = after ? hint.after_want_head : hint.want_head;
      next.push_back({.slot = pending[j].slot,
                      .stage = 0,
                      .phase = pl::Phase::kPrefill,
                      .past = unit.past + unit.rows + (after ? hint.rows : 0),
                      .rows = rows,
                      .outputs = head ? 1U : 0U,
                      .want_head = head,
                      .greedy = false,
                      .row_invariant = false,
                      .alone = false});
    }
    std::vector<pl::StepShape> steps;
    if (next.empty()) return steps;
    const bool join = std::ranges::all_of(next, [&](const en::PendingWork& w) {
      return CompatiblePrefill(next.front().past, next.front().rows, w.past, w.rows);
    });
    const auto compose = [&](std::span<const en::PendingWork> work) {
      en::Composition composed = composer_->Compose(work);
      if (!composed.refused.empty()) return false;
      for (en::ComposedStep& c : composed.steps) steps.push_back(std::move(c.shape));
      return true;
    };
    if (join) {
      if (!compose(next)) steps.clear();
      return steps;
    }
    for (const en::PendingWork& w : next) {
      if (!compose(std::span(&w, 1))) return {};
    }
    return steps;
  };
  const en::AheadFor ahead = [&](const en::ComposedStep& /*step*/,
                                 std::span<const std::size_t> owners) {
    en::StepAhead a;
    if (prefill_.lookahead == 0 && !prefill_.capture_ahead && !prefill_.prepare_ahead) return a;
    std::size_t filled = 0;
    const auto push_shape = [&](pl::StepShape&& s) {
      if (filled < a.shapes.size()) a.shapes[filled++] = std::move(s);
    };
    const bool row_invariant = speculative_ && speculative_->mode().row_invariant;
    const auto push_verify_futures = [&](std::size_t j, std::uint32_t end_past) {
      const PreparedPrefill& unit = prepared[of[j]];
      const bool greedy = device_verify_ && unit.branch != nullptr && !sampling(*unit.branch);
      const std::uint32_t min_vrows = branch_depth_ ? 1U : verify_rows_;
      for (std::uint32_t vrows = verify_rows_; vrows >= min_vrows && filled < a.shapes.size();
           --vrows) {
        if (end_past > context_ || vrows > context_ - end_past) continue;
        const en::PendingWork vw{.slot = pending[j].slot,
                                 .stage = 0,
                                 .phase = pl::Phase::kVerify,
                                 .past = end_past,
                                 .rows = vrows,
                                 .outputs = vrows,
                                 .want_head = true,
                                 .greedy = greedy,
                                 .row_invariant = row_invariant,
                                 .alone = false};
        en::Composition composed = composer_->Compose(std::span(&vw, 1));
        if (composed.refused.empty() && composed.steps.size() == 1) {
          push_shape(std::move(composed.steps.front().shape));
        }
      }
      if (filled < a.shapes.size() && verify_rows_ > 1 && end_past < context_ &&
          verify_rows_ <= context_ - end_past) {
        const std::uint32_t align = std::max<std::uint32_t>(caps_.joined_read_align, 256U);
        const std::uint32_t next_bucket_past =
            ((end_past + verify_rows_ + align - 1U) / align) * align;
        const std::uint32_t later_past =
            std::min(std::max(end_past + 128U, next_bucket_past), context_ - verify_rows_);
        if (later_past > end_past) {
          const en::PendingWork vw{.slot = pending[j].slot,
                                   .stage = 0,
                                   .phase = pl::Phase::kVerify,
                                   .past = later_past,
                                   .rows = verify_rows_,
                                   .outputs = verify_rows_,
                                   .want_head = true,
                                   .greedy = greedy,
                                   .row_invariant = row_invariant,
                                   .alone = false};
          en::Composition composed = composer_->Compose(std::span(&vw, 1));
          if (composed.refused.empty() && composed.steps.size() == 1) {
            push_shape(std::move(composed.steps.front().shape));
          }
        }
      }
    };
    const auto push_post_prefill_futures = [&](std::size_t j, std::uint32_t end_past) {
      const PreparedPrefill& unit = prepared[of[j]];
      bool plain_multi = false;
      std::uint32_t solo_tail_past = end_past;
      if (wave_mode_ && active_branches_.size() > 1) {
        std::vector<en::PendingWork> decode_work;
        std::vector<std::uint32_t> generated;
        for (Branch* b : active_branches_) {
          if (b == nullptr) continue;
          const bool is_cur = (b == unit.branch);
          const std::uint32_t slot = Slot(*b);
          const std::uint32_t done_pos = prefill_done_pos_.at(slot);
          if (!is_cur && done_pos == 0) continue;
          const std::uint32_t pos = is_cur ? end_past : CursorFor(*b);
          if (pos == 0 || pos >= context_ || (!is_cur && pos < done_pos)) continue;
          const bool greedy = device_greedy_ && !sampling(*b);
          decode_work.push_back({.slot = slot,
                                 .stage = 0,
                                 .phase = pl::Phase::kDecode,
                                 .past = pos,
                                 .rows = 1,
                                 .outputs = 1,
                                 .want_head = true,
                                 .greedy = greedy,
                                 .row_invariant = row_invariant,
                                 .alone = false});
          generated.push_back(is_cur ? 0U : (pos - done_pos));
        }
        if (decode_work.size() >= 2 &&
            wave_mode_->Choose(static_cast<std::uint32_t>(decode_work.size())) ==
                execution::AdaptiveWaveMode::Mode::kPlain) {
          plain_multi = true;
          const auto push_composed = [&](std::span<const en::PendingWork> dw) {
            if (filled >= a.shapes.size()) return;
            en::Composition composed = composer_->Compose(dw);
            if (composed.refused.empty() && composed.steps.size() == 1) {
              push_shape(std::move(composed.steps.front().shape));
            }
          };
          // 1. Current multi-slot wave at d = 0.
          push_composed(decode_work);
          // 2. Future multi-slot waves advancing past compressor boundaries.
          for (const std::uint32_t d : {64U, 128U}) {
            std::vector<en::PendingWork> advanced = decode_work;
            for (auto& w : advanced) w.past = std::min(w.past + d, context_ - 1U);
            push_composed(advanced);
          }
          // 3. Tail waves as peers finish in descending order of already-generated tokens.
          std::vector<std::size_t> peer_order;
          for (std::size_t idx = 0; idx < decode_work.size(); ++idx) {
            if (decode_work[idx].slot != Slot(*unit.branch)) peer_order.push_back(idx);
          }
          std::ranges::stable_sort(peer_order, [&](std::size_t a_idx, std::size_t b_idx) {
            return generated[a_idx] > generated[b_idx];
          });
          std::vector<bool> dropped(decode_work.size(), false);
          for (std::size_t p_i = 0; p_i < peer_order.size(); ++p_i) {
            const std::size_t drop_idx = peer_order[p_i];
            dropped[drop_idx] = true;
            const std::uint32_t gen = generated[drop_idx];
            const std::uint32_t rem = gen < 128U ? 128U - gen : 1U;
            solo_tail_past = std::min(end_past + rem,
                                      context_ > verify_rows_ ? context_ - verify_rows_ : end_past);
            std::vector<en::PendingWork> surviving;
            for (std::size_t idx = 0; idx < decode_work.size(); ++idx) {
              if (dropped[idx]) continue;
              en::PendingWork w = decode_work[idx];
              w.past = std::min(w.past + rem, context_ - 1U);
              surviving.push_back(w);
            }
            if (!surviving.empty()) {
              push_composed(surviving);
            }
          }
        }
      }
      if (speculate_ && verify_rows_ > 1) {
        push_verify_futures(j, plain_multi ? solo_tail_past : end_past);
      }
    };
    std::vector<pl::StepShape> near = future(owners, false);
    std::vector<pl::StepShape> far;
    if (near.size() == 1) {
      push_shape(std::move(near.front()));
      far = future(owners, true);
      if (far.size() == 1) {
        push_shape(std::move(far.front()));
      } else if (far.empty() && owners.size() == 1) {
        const std::size_t j = owners.front();
        const PreparedPrefill& unit = prepared[of[j]];
        if (unit.past > 0 || unit.next.rows > 0) {
          push_post_prefill_futures(j, unit.past + unit.rows + unit.next.rows);
        }
      }
    } else if (near.size() > 1) {
      // The owners' next chunks step apart (a prompt's last chunk is
      // shorter than its peer's): the first two of those steps, each built
      // while this one runs rather than on its own first run, when the
      // device would wait for it.
      push_shape(std::move(near[0]));
      push_shape(std::move(near[1]));
    } else if (near.empty() && owners.size() == 1) {
      const std::size_t j = owners.front();
      const PreparedPrefill& unit = prepared[of[j]];
      if (unit.past > 0 || unit.rows > 0) {
        push_post_prefill_futures(j, unit.past + unit.rows);
      }
    } else if (near.empty() && owners.size() > 1) {
      const bool plain_wave =
          !speculate_ ||
          (wave_mode_ && wave_mode_->Choose(static_cast<std::uint32_t>(owners.size())) ==
                             execution::AdaptiveWaveMode::Mode::kPlain);
      if (plain_wave) {
        std::vector<en::PendingWork> decode_work;
        for (const std::size_t j : owners) {
          const PreparedPrefill& unit = prepared[of[j]];
          const std::uint32_t end_past = unit.past + unit.rows;
          if (end_past >= context_) continue;
          const bool greedy = device_greedy_ && unit.branch != nullptr && !sampling(*unit.branch);
          decode_work.push_back({.slot = pending[j].slot,
                                 .stage = 0,
                                 .phase = pl::Phase::kDecode,
                                 .past = end_past,
                                 .rows = 1,
                                 .outputs = 1,
                                 .want_head = true,
                                 .greedy = greedy,
                                 .row_invariant = row_invariant,
                                 .alone = false});
        }
        if (decode_work.size() == owners.size()) {
          en::Composition composed = composer_->Compose(decode_work);
          if (composed.refused.empty() && composed.steps.size() == 1) {
            push_shape(std::move(composed.steps.front().shape));
          }
        }
      }
    }
    a.build =
        ((near.empty() || far.empty()) && speculate_ && verify_rows_ > 1 && prefill_.lookahead > 0)
            ? 4U
            : std::min<std::uint32_t>(prefill_.lookahead, 2);
    a.capture_ahead = !near.empty() && prefill_.capture_ahead;
    a.prepare = !near.empty() && prefill_.prepare_ahead;
    return a;
  };
  if (!pending.empty()) {
    std::vector<std::span<const std::int32_t>> histories;
    for (const auto j : of) histories.push_back(prepared[j].all);
    const en::WaveReport wave =
        en::RunPromptWave(*runner_, *composer_, pending, work, histories, ahead);
    for (std::size_t j = 0; j < pending.size(); ++j) {
      PreparedPrefill& unit = prepared[of[j]];
      const en::WaveOutcome& ran = wave.outcomes[j];
      if (ran.refusal) {
        unit.result = Error(ran.refusal->what);
        continue;
      }
      if (unit.next.rows == 0) {
        prefill_done_pos_.at(Slot(*unit.branch)) = unit.past + unit.rows;
      } else {
        prefill_done_pos_.at(Slot(*unit.branch)) = 0;
      }
      ResetDrafters(pending[j].slot);
      if (unit.want_head) {
        auto row = LogitsRow(ran.outputs, 0);
        if (!row) {
          unit.result = Error(row.error());
          continue;
        }
        if (unit.logits != nullptr) *unit.logits = std::move(*row);
      } else if (unit.logits != nullptr) {
        unit.logits->clear();
      }
      unit.result = {};
    }
    if (pending.size() > 1 && wave.steps < pending.size()) {
      ++counts_.joined_prefill_steps;
      for (const en::PendingWork& p : pending) counts_.joined_prefill_rows += p.rows;
    }
  }
  return GenerationCohortUsable() ? Status{}
                                  : Error(std::format("{}'s request cohort faulted", name_));
}

Status EngineLlm::RunPreparedPrefillWave(std::span<PreparedPrefill> prepared) {
  if (prepared.empty() || prepared.size() > prefill_wave_capacity()) {
    return Error(std::format("{}'s prompt wave exceeds its capability", name_));
  }
  return RunPrefill(prepared);
}

Status EngineLlm::RunPrefillChunkFor(Branch& branch, std::span<const std::int32_t> all,
                                     std::uint32_t past, bool /*inject*/, bool want_head,
                                     std::vector<float>& logits, PrefillHint next) {
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

Status EngineLlm::PreparePrefillStateFor(Branch& branch, std::uint32_t past, std::uint32_t rows) {
  if (auto built = Built(); !built) return built;
  const std::uint32_t slot = Slot(branch);
  if (past != runner_->slots().positions(slot) || rows == 0 || past > context_ ||
      rows > context_ - past) {
    return Error(std::format("{}'s prefill funding needs an owned bounded continuation", name_));
  }
  return runner_->slots().ReserveStateThrough(slot, past + rows);
}

Status EngineLlm::RunChunkFor(Branch& branch, std::span<const std::int32_t> all, std::uint32_t past,
                              bool /*inject*/, std::vector<float>& logits) {
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

std::optional<Status> EngineLlm::RunGreedyChunkFor(Branch& branch,
                                                   std::span<const std::int32_t> all,
                                                   std::uint32_t past, std::int32_t& token) {
  if (!device_greedy_) return std::nullopt;
  if (auto built = Built(); !built) return built;
  if (all.size() != std::size_t{past} + 1) {
    return Error(std::format("{}'s greedy step needs one anchor row", name_));
  }
  // A greedy device choice regardless of the branch's sampling: the caller
  // asks only for greedy rows without a caller of their logits.
  const std::uint32_t slot = Slot(branch);
  std::array<en::DecodeWork, 1> work{
      en::DecodeWork{.slot = slot, .history = all, .left = 1, .greedy = true, .score = {}}};
  const en::DecodeReport report = plain_->Step(work);
  const en::DecodeOutcome& out = report.outcomes.front();
  if (out.refusal) return Error(out.refusal->what);
  if (out.outputs.size() != sizeof(token)) {
    return Error(std::format("{}'s greedy step returned no token", name_));
  }
  std::memcpy(&token, out.outputs.data(), sizeof(token));
  if (token < 0 || std::cmp_greater_equal(token, logit_width_)) {
    return Error(std::format("{}'s greedy step chose token {}", name_, token));
  }
  ++counts_.greedy_tokens;
  return Status{};
}

Status EngineLlm::RunChunk(std::span<const std::int32_t> all, std::uint32_t past, bool inject,
                           std::vector<float>& logits) {
  return RunChunkFor(default_branch(), all, past, inject, logits);
}

// ------------------------------------------------------------- the state

Status EngineLlm::SettleFor(Branch& branch) {
  if (auto built = Built(); !built) return built;
  const std::uint32_t slot = Slot(branch);
  en::SlotLifecycle& slots = runner_->slots();
  if (!slots.spilled(slot) && slots.owed(slot)) {
    if (auto settled = slots.Settle(slot); !settled) return settled;
  }
  return slots.usable(slot) ? Status{}
                            : Error(std::format("{}'s conversation state is unusable", name_));
}

Status EngineLlm::ClearStateFor(Branch& branch) {
  if (auto built = Built(); !built) return built;
  if (branch_depth_)
    BranchDecoding(branch) = execution::AdaptiveDepth(branch_depth_, depth_cost_ratio_);
  prefill_done_pos_.at(Slot(branch)) = 0;
  return runner_->slots().Clear(Slot(branch));
}

bool EngineLlm::StateUsableFor(const Branch& branch) const {
  // Prefix reuse includes healthy kept state that still needs hydration.
  // Execution guards continue to reject its pending progress replacement.
  return runner_ &&
         (runner_->slots().usable(Slot(branch)) || runner_->slots().restorable(Slot(branch)));
}

bool EngineLlm::StateRefusedFor(const Branch& branch) const {
  return runner_ && runner_->slots().state_refused(Slot(branch));
}

Status EngineLlm::ReleaseIdleStateFor(Branch& branch) {
  if (auto built = Built(); !built) return built;
  if (branch_depth_)
    BranchDecoding(branch) = execution::AdaptiveDepth(branch_depth_, depth_cost_ratio_);
  prefill_done_pos_.at(Slot(branch)) = 0;
  return runner_->slots().ClearIdle(Slot(branch));
}

Status EngineLlm::SpillFor(Branch& branch) {
  if (auto built = Built(); !built) return built;
  prefill_done_pos_.at(Slot(branch)) = 0;
  return runner_->slots().Spill(Slot(branch));
}

Status EngineLlm::RestoreFor(Branch& branch) {
  if (auto built = Built(); !built) return built;
  return runner_->slots().Restore(Slot(branch));
}

bool EngineLlm::SpilledFor(const Branch& branch) const {
  return runner_ && runner_->slots().spilled(Slot(branch));
}

std::uint64_t EngineLlm::SpilledBytesFor(const Branch& branch) const {
  return runner_ ? runner_->slots().spilled_bytes(Slot(branch)) : 0;
}

bool EngineLlm::LeasedFor(const Branch& branch) const {
  return runner_ && runner_->slots().held(Slot(branch));
}

std::uint64_t EngineLlm::RefusedBytesFor(const Branch& branch) const {
  if (!runner_) return 0;
  const en::SlotLifecycle& slots = runner_->slots();
  const std::uint32_t slot = Slot(branch);
  // A refused restore asks for what the slot's files hold.
  return slots.spilled(slot) ? slots.spilled_bytes(slot) : slots.refused_bytes(slot);
}

std::uint64_t EngineLlm::SpillWriteBytesFor(const Branch& branch) const {
  return runner_ ? runner_->slots().spill_write_bytes(Slot(branch)) : 0;
}

Status EngineLlm::PrepareDecodeStateFor(Branch& branch, std::uint32_t pos, std::uint32_t left) {
  if (auto built = Built(); !built) return built;
  const std::uint32_t slot = Slot(branch);
  if (pos != runner_->slots().positions(slot) || pos >= context_) {
    return Error(std::format("{}'s decode position differs from its completed state", name_));
  }
  // A verify's rows (the anchor and every proposal) at most, within the
  // context: the decoding orchestrator never takes more.
  const std::uint32_t rows =
      speculate_ ? std::max<std::uint32_t>(std::min({left, verify_rows_, context_ - pos}), 1) : 1;
  return runner_->slots().ReserveStateThrough(slot, pos + rows);
}

std::uint64_t EngineLlm::TargetStateBaseFor(const Branch& branch) const {
  return runner_ ? runner_->slots().live(Slot(branch), 0).base(0) : 0;
}
std::uint64_t EngineLlm::TargetStateBytesFor(const Branch& branch) const {
  return runner_ ? runner_->slots().live(Slot(branch), 0).bytes(0) : 0;
}
std::uint64_t EngineLlm::DrafterStateBaseFor(const Branch& branch) const {
  return runner_ && runner_->slots().stages() > 1 ? runner_->slots().live(Slot(branch), 1).base(0)
                                                  : 0;
}
std::uint64_t EngineLlm::DrafterStateBytesFor(const Branch& branch) const {
  return runner_ && runner_->slots().stages() > 1 ? runner_->slots().live(Slot(branch), 1).bytes(0)
                                                  : 0;
}
std::uint64_t EngineLlm::UsedStateBytesFor(const Branch& branch) const {
  return runner_ ? runner_->slots().used_bytes(Slot(branch)) : 0;
}

std::vector<en::LiveState::Range> EngineLlm::UsedStateRangesFor(const Branch& branch) const {
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

Status EngineLlm::SaveUsedStateFor(Branch& branch, void* host,
                                   std::span<const en::LiveState::Range> ranges) {
  if (auto built = Built(); !built) return built;
  return runner_->slots().CopyState(Slot(branch), host, ranges, /*to_host=*/true);
}

Status EngineLlm::RestoreUsedStateFor(Branch& /*branch*/, void* /*host*/,
                                      std::span<const en::LiveState::Range> /*ranges*/) {
  return Error(std::format("{} restores its state only with its positions", name_));
}

std::expected<std::vector<en::LiveState::Range>, std::string> EngineLlm::CheckpointRangesFor(
    const Branch& branch, std::uint32_t positions) const {
  if (auto built = Built(); !built) return std::unexpected(built.error());
  const std::uint32_t slot = Slot(branch);
  if (positions != runner_->slots().positions(slot)) {
    return Error(std::format("{}'s checkpoint boundary differs from its completed state", name_));
  }
  return runner_->slots().CheckpointRanges(slot, positions);
}

Status EngineLlm::PrepareRestoreStateFor(Branch& /*branch*/,
                                         std::span<const en::LiveState::Range> /*footprint*/) {
  return Error(std::format("{} restores its state only with its positions", name_));
}

Status EngineLlm::CopyCheckpointStateFor(Branch& branch, void* host,
                                         std::span<const en::LiveState::Range> ranges,
                                         bool to_host) {
  if (auto built = Built(); !built) return built;
  return runner_->slots().CopyState(Slot(branch), host, ranges, to_host);
}

Status EngineLlm::CheckCheckpointMetadataFor(
    const Branch& /*branch*/, std::uint32_t position, std::uint32_t cursor,
    std::span<const en::LiveState::Range> footprint) const {
  if (auto built = Built(); !built) return built;
  if (position != cursor) {
    return Error(std::format("{}'s checkpoint cursor differs from its positions", name_));
  }
  return runner_->slots().ValidateFootprint(position, footprint);
}

Status EngineLlm::PreparePositionRestoreFor(Branch& branch, std::uint32_t position,
                                            std::uint32_t cursor,
                                            std::span<const en::LiveState::Range> footprint,
                                            std::string_view layout) {
  if (auto checked = CheckCheckpointMetadataFor(branch, position, cursor, footprint); !checked) {
    return checked;
  }
  prefill_done_pos_.at(Slot(branch)) = 0;
  return runner_->slots().PrepareRestore(Slot(branch), position, footprint, layout);
}

Status EngineLlm::CompletePositionRestoreFor(Branch& branch, std::uint32_t position) {
  if (auto built = Built(); !built) return built;
  if (auto completed = runner_->slots().CompleteRestore(Slot(branch), position); !completed) {
    return completed;
  }
  prefill_done_pos_.at(Slot(branch)) = position;
  ResetDrafters(Slot(branch));
  return {};
}

Status EngineLlm::RestoreSnapshotFor(Branch& branch, void* host,
                                     std::span<const en::LiveState::Range> ranges,
                                     std::uint32_t cursor, std::string_view layout) {
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

Status EngineLlm::AdoptPositionFor(Branch& branch, std::span<const std::int32_t> tokens,
                                   std::uint32_t cursor,
                                   std::span<const en::LiveState::Range> ranges,
                                   std::string_view layout) {
  if (auto built = Built(); !built) return built;
  if (cursor == 0 || cursor != tokens.size()) {
    return Error(std::format("{}'s kept positions differ from its token history", name_));
  }
  if (auto adopted = runner_->slots().Adopt(Slot(branch), cursor, ranges, layout); !adopted) {
    return adopted;
  }
  prefill_done_pos_.at(Slot(branch)) = cursor;
  ResetDrafters(Slot(branch));
  return {};
}

std::uint32_t EngineLlm::CursorFor(const Branch& branch) const {
  return runner_ ? runner_->slots().positions(Slot(branch)) : 0;
}

}  // namespace llmp::runtime
