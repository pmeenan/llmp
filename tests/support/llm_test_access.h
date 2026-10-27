// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_TESTS_SUPPORT_LLM_TEST_ACCESS_H_
#define LLMP_TESTS_SUPPORT_LLM_TEST_ACCESS_H_

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "engine/decoding.h"
#include "runtime/serving.h"

namespace llmp::runtime {

// Fixed-size digests keep observation bounded; callback-owned output spans are
// consumed before the original policy callback returns to the orchestrator.
struct LlmDecodeObservation {
  static constexpr std::size_t kMaxEvents = 512;
  static constexpr std::size_t kMaxTokensPerEvent = 16;
  enum class Kind : std::uint8_t { kDeviceTokens, kHeadRows };
  struct Event {
    bool verify = false;
    std::uint32_t slot = 0;
    std::uint32_t ordinal = 0;
    std::uint32_t requested_rows = 0;
    std::uint32_t actual_rows = 0;
    std::uint32_t head_width = 0;
    std::size_t drafter = 0;
    std::uint16_t stage = 0;
    Kind kind = Kind::kHeadRows;
    std::string head_map;
    std::string raw_output;
    std::string head_values;
    std::vector<std::uint32_t> proposal_rows;
    std::vector<std::int32_t> input_tokens;
    std::vector<std::int32_t> output_tokens;
    std::vector<std::int32_t> result_tokens;
    std::uint32_t keep = 0;
    bool operator==(const Event&) const = default;
  };
  static constexpr std::uint64_t BoundedBytes(std::size_t limit, std::size_t token_limit) {
    // Include the outer vector/object, full event storage, per-allocation
    // slack for three strings and four bounded vectors, and a bounded text
    // serialization of the events.
    return 128U + std::uint64_t{kMaxEvents} * sizeof(Event) +
           std::uint64_t{limit} *
               (3U * 64U + token_limit * (3U * sizeof(std::int32_t) + sizeof(std::uint32_t)) +
                6U * 64U + 1024U);
  }

  explicit LlmDecodeObservation(std::size_t requested_limit = kMaxEvents,
                                std::size_t requested_token_limit = kMaxTokensPerEvent)
      : limit(std::min(requested_limit, kMaxEvents)),
        token_limit(std::min(requested_token_limit, kMaxTokensPerEvent)),
        valid(requested_limit != 0 && requested_limit <= kMaxEvents && requested_token_limit != 0 &&
              requested_token_limit <= kMaxTokensPerEvent) {
    events.resize(limit);
    for (auto& event : events) {
      event.head_map.reserve(64);
      event.raw_output.reserve(64);
      event.head_values.reserve(64);
      event.proposal_rows.reserve(token_limit);
      event.input_tokens.reserve(token_limit);
      event.output_tokens.reserve(token_limit);
      event.result_tokens.reserve(token_limit);
    }
  }
  std::size_t limit;
  std::size_t token_limit;
  std::size_t count = 0;
  bool incomplete = false;
  bool valid;
  std::vector<Event> events;
  std::array<std::uint32_t, 16> ordinals{};
};

// Test-only access to host configuration, completed session facts, and the
// private production decoding policy. Native work and failures still run
// through the runner, source and device providers.
struct LlmTestAccess {
  static void ObserveDepth(Llm& model, Llm::Branch& branch, std::uint32_t depth,
                           std::uint32_t kept) {
    model.BranchDecoding(branch).Observe(depth, kept);
  }
  static const execution::AdaptiveDepth& Depth(const Llm& model, const Llm::Branch& branch) {
    return model.BranchDecoding(branch);
  }
  static Status UseChat(Llm& model, tokenizer::Tokenizer tokenizer, std::string_view text) {
    model.tokenizer_ = std::make_unique<tokenizer::Tokenizer>(std::move(tokenizer));
    return model.UseTemplate(text);
  }
  static Status UseChatAssets(Llm& model, const artifact::Artifact& artifact,
                              const config::ModelEntry& entry, std::uint64_t cap) {
    return model.UseChatAssets(artifact, entry, cap);
  }
  static bool HasChat(const Llm& model) { return model.tokenizer_ && model.template_; }
  // The drafter taps' bookkeeping of the branch's slot now, and what its
  // `index`th turn checkpoint recorded.
  static engine::SlotTapRows Taps(const Llm& model, const Llm::Branch& branch) {
    return model.runner().SaveTapRows(model.Slot(branch));
  }
  static Status SetTaps(Llm& model, const Llm::Branch& branch, const engine::SlotTapRows& rows) {
    return model.runner().RestoreTapRows(model.Slot(branch), rows);
  }
  static engine::SlotTapRows CheckpointTaps(const Llm::Branch& branch, std::size_t index) {
    return branch.turn_checkpoints_.at(index).taps;
  }
  static auto ChatPool(const Llm& model) { return model.chat_startup_; }
  static const chat::ChatTemplate& ChatTemplate(const Llm& model) { return *model.template_; }
  static std::size_t LongestToken(const Llm& model) { return model.longest_token_; }
  static PrefillHint Hint(const Llm::PromptSession& session, std::uint32_t current_end) {
    return session.NextPrefillHint(current_end);
  }
  static Status ExecuteChunk(Llm& model, Llm::Branch& branch,
                             const Llm::GenerationSession::Step& step, std::vector<float>& row) {
    return model.RunChunkFor(branch, step.all, step.position, row);
  }
  static void Stops(Llm& model, std::vector<std::int32_t> stops) {
    model.stops_ = std::move(stops);
  }
  // A verdict from the model's decoding policy, as a verify's completion asks
  // for it: the slot's verify returned device tokens (`device`) or rows.
  static std::expected<engine::Verdict, std::string> Judge(Llm& model, std::uint32_t slot,
                                                           bool device,
                                                           std::span<const std::int32_t> verified,
                                                           std::span<const std::byte> outputs) {
    model.verify_tokens_.at(slot) = device;
    return model.Policy().judge(slot, verified, outputs);
  }
  // The row a decoding policy's readback names after a greedy step of
  // `tokens` whose device choices are `chosen` (engine::DecodingPolicy::
  // readback), and a read-back row the slot's next judge sees.
  static std::optional<std::uint32_t> Readback(Llm& model, std::uint32_t slot,
                                               std::span<const std::int32_t> tokens,
                                               std::span<const std::int32_t> chosen) {
    return model.Policy().readback(slot, tokens, std::as_bytes(chosen));
  }
  static void ReadRow(Llm& model, std::uint32_t slot, std::uint32_t row,
                      std::vector<float> logits) {
    model.read_rows_.at(slot) = Llm::HeadRow{.row = row, .logits = std::move(logits)};
  }
  static void ClearRow(Llm& model, std::uint32_t slot) { model.read_rows_.at(slot).reset(); }
  static void UseTools(Llm::Branch& branch, chat::ToolSampling* sampling) {
    branch.tool_sampling_ = sampling;
  }
  static void UseSampling(Llm::Branch& branch, std::optional<execution::SamplingParams> sampling,
                          std::uint64_t seed = 0) {
    branch.sampling_ = sampling;
    branch.seed_ = seed;
  }
  // The lazy top-k's compacted tokens (Llm::LeadingAllowed), in ID order.
  static std::optional<std::vector<std::int32_t>> LeadingAllowed(Llm& model, Llm::Branch& branch,
                                                                 std::span<const float> row,
                                                                 const chat::ToolSampling& mask,
                                                                 std::size_t k) {
    if (!model.LeadingAllowed(branch, row, mask, k)) return std::nullopt;
    return branch.lazy_ids_;
  }
  static std::expected<std::int32_t, std::string> Choose(Llm& model, Llm::Branch& branch,
                                                         std::span<const float> row,
                                                         std::uint64_t position) {
    return model.Choose(branch, row, position);
  }
  // A speculative verdict on `draft` (Llm::Keep): accepted, and the token
  // kept in `next`.
  static std::expected<bool, std::string> Keep(Llm& model, Llm::Branch& branch,
                                               std::span<const float> row, std::int32_t draft,
                                               std::uint64_t position, std::int32_t& next) {
    return model.Keep(branch, row, draft, position, next);
  }
  // How many logits the branch's last constrained choice held: the whole
  // row once masked, the kept tokens alone when chosen lazily.
  static std::size_t ConstrainedLogits(const Llm::Branch& branch) {
    return branch.constrained_logits_.size();
  }
  static bool Ready(const Llm& model, const Llm::Branch& branch) {
    return model.GenerationReadyFor(branch);
  }
  static std::optional<std::uint32_t> Remaining(const Llm& model, const Llm::Branch& branch) {
    return model.RemainingGenerationRowsFor(branch);
  }

  // A full permutation exercises selected-head metadata without changing the
  // fake runner's output width or using synthetic production policy state.
  static Status SetMappedDrafterHeadForTest(Llm& model, std::size_t drafter) {
    if (!model.speculative_ || drafter >= model.drafter_specs_.size())
      return std::unexpected("mapped-head diagnostic names an absent drafter");
    const auto& counts = model.counts_;
    if (counts.joined_prefill_steps != 0 || counts.joined_causal_steps != 0 ||
        counts.speculative_units != 0 || counts.drafted != 0 || counts.accepted != 0)
      return std::unexpected("mapped head must be configured before model work");
    auto mode = model.speculative_->mode();
    if (drafter >= mode.drafters.size())
      return std::unexpected("mapped-head mode has no matching drafter");
    std::vector<std::int32_t> ids(model.logit_width_);
    for (std::size_t row = 0; row < ids.size(); ++row)
      ids[row] = static_cast<std::int32_t>(ids.size() - row - 1);
    auto head_ids = std::make_shared<const std::vector<std::int32_t>>(std::move(ids));
    mode.drafters[drafter].head_ids = head_ids;
    auto rebuilt = engine::DecodingOrchestrator::Make(*model.runner_, *model.composer_,
                                                      std::move(mode), model.Policy());
    if (!rebuilt) return std::unexpected(rebuilt.error());
    model.drafter_specs_[drafter].head_ids = std::move(head_ids);
    model.speculative_.emplace(std::move(*rebuilt));
    return {};
  }

  // Install only before model work. A diagnostic may request the existing
  // full-head drafter output form, but cannot reset cursors or adaptation.
  static Status InstallDecodeObservation(Llm& model, LlmDecodeObservation& observation,
                                         bool full_head) {
    if (!model.runner_ || !model.composer_ || !model.speculative_)
      return std::unexpected("model has no configured speculative policy");
    const auto& counts = model.counts_;
    if (counts.joined_prefill_steps != 0 || counts.joined_prefill_rows != 0 ||
        counts.joined_causal_steps != 0 || counts.joined_causal_rows != 0 ||
        counts.joined_waves != 0 || counts.joined_units != 0 || counts.speculative_units != 0 ||
        counts.drafted != 0 || counts.accepted != 0 || counts.discarded != 0 ||
        counts.greedy_tokens != 0)
      return std::unexpected("observation must be installed before any model work");
    const auto empty_branch = [&model](const Llm::Branch& branch) {
      return &branch.model() == &model && branch.history().empty() &&
             branch.turn_checkpoints() == 0 && !branch.HeldByContinuation() &&
             model.BranchIdle(branch);
    };
    if (!empty_branch(model.default_branch_))
      return std::unexpected("observation requires an empty default branch");
    for (std::uint32_t slot = 0; slot < model.slots_; ++slot) {
      const auto& slots = model.runner_->slots();
      if (slots.positions(slot) != 0 || slots.spilled(slot) || slots.restorable(slot))
        return std::unexpected("observation must be installed before state or restoration");
      for (std::size_t stage = 0; stage < slots.stages(); ++stage)
        if (!slots.live(slot, stage).used_ranges().empty())
          return std::unexpected("observation must be installed before state writes");
    }
    for (const auto& branch : model.extra_branches_)
      if (branch && !empty_branch(*branch))
        return std::unexpected("observation requires empty, inactive branches");
    if (!observation.valid || observation.count != 0 || observation.incomplete ||
        observation.events.size() != observation.limit || observation.token_limit == 0)
      return std::unexpected("invalid bounded observation capacity");

    auto mode = model.speculative_->mode();
    if (mode.drafters.size() != model.drafter_specs_.size() || mode.drafters.empty())
      return std::unexpected("the configured drafter policy is inconsistent");
    if (mode.verify_rows() > observation.token_limit)
      return std::unexpected("bounded observation token capacity is below verify width");
    for (const auto& drafter : mode.drafters)
      if (drafter.max_depth > observation.token_limit)
        return std::unexpected("bounded observation token capacity is below drafter depth");
    if (full_head) {
      for (auto& drafter : mode.drafters) drafter.device_tokens = false;
    }

    auto policy = model.Policy();
    auto propose = std::move(policy.propose);
    auto judge = std::move(policy.judge);
    policy.propose =
        [&model, &observation, propose = std::move(propose)](
            std::uint32_t slot, std::size_t drafter, std::span<const std::byte> outputs,
            std::uint32_t want) mutable -> std::expected<std::vector<std::int32_t>, std::string> {
      auto result = propose(slot, drafter, outputs, want);
      if (observation.count >= observation.limit || slot >= observation.ordinals.size() ||
          drafter >= model.drafter_specs_.size() || want > observation.token_limit) {
        observation.incomplete = true;
        return result;
      }
      const auto& spec = model.drafter_specs_[drafter];
      auto& event = observation.events[observation.count++];
      event.slot = slot;
      event.ordinal = observation.ordinals[slot]++;
      event.requested_rows = want;
      event.drafter = drafter;
      event.stage = spec.stage;
      event.kind = spec.device_tokens ? LlmDecodeObservation::Kind::kDeviceTokens
                                      : LlmDecodeObservation::Kind::kHeadRows;
      if (spec.head_ids) {
        base::Sha256 map_hash;
        map_hash.Update(std::as_bytes(std::span(*spec.head_ids)));
        event.head_map = base::ToHex(map_hash.Finish());
      }
      base::Sha256 raw_hash;
      raw_hash.Update(outputs);
      event.raw_output = base::ToHex(raw_hash.Finish());
      if (result) {
        if (result->size() > observation.token_limit) {
          observation.incomplete = true;
          return result;
        }
        event.result_tokens.assign(result->begin(), result->end());
        event.actual_rows = static_cast<std::uint32_t>(result->size());
      }
      if (spec.device_tokens) {
        // The production policy has already validated native vocabulary IDs.
      } else {
        auto width = engine::DrafterHeadRows(spec, model.logit_width_);
        if (!width) {
          observation.incomplete = true;
          return result;
        }
        event.head_width = *width;
        const std::size_t row_bytes = std::size_t{*width} * sizeof(float);
        const std::size_t rows =
            row_bytes == 0 ? 0 : std::min(outputs.size() / row_bytes, std::size_t{want});
        event.actual_rows = static_cast<std::uint32_t>(rows);
        if (result && result->size() != rows) observation.incomplete = true;
        const auto bytes = outputs.first(rows * row_bytes);
        const auto* values = reinterpret_cast<const float*>(bytes.data());
        if (!std::ranges::all_of(std::span(values, rows * *width),
                                 [](float value) { return std::isfinite(value); }))
          observation.incomplete = true;
        for (std::size_t row = 0; row < rows; ++row) {
          const auto logits = std::span(values + row * *width, *width);
          const auto index =
              static_cast<std::uint32_t>(std::ranges::max_element(logits) - logits.begin());
          event.proposal_rows.push_back(index);
          const auto mapped = engine::DrafterHeadToken(spec, index, model.logit_width_);
          if (!mapped || !result || row >= result->size() || (*result)[row] != *mapped)
            observation.incomplete = true;
        }
        base::Sha256 head_hash;
        head_hash.Update(bytes);
        event.head_values = base::ToHex(head_hash.Finish());
      }
      return result;
    };
    policy.judge = [&model, &observation, judge = std::move(judge)](
                       std::uint32_t slot, std::span<const std::int32_t> verified,
                       std::span<const std::byte> outputs) mutable
        -> std::expected<engine::Verdict, std::string> {
      auto result = judge(slot, verified, outputs);
      if (observation.count >= observation.limit || slot >= observation.ordinals.size() ||
          verified.size() > observation.token_limit) {
        observation.incomplete = true;
        return result;
      }
      auto& event = observation.events[observation.count++];
      event.verify = true;
      event.slot = slot;
      event.ordinal = observation.ordinals[slot]++;
      event.requested_rows = static_cast<std::uint32_t>(verified.size());
      event.actual_rows = static_cast<std::uint32_t>(verified.size());
      event.head_width = model.logit_width_;
      event.input_tokens.assign(verified.begin(), verified.end());
      if (slot >= model.verify_tokens_.size()) {
        observation.incomplete = true;
        return result;
      }
      event.kind = model.verify_tokens_[slot] ? LlmDecodeObservation::Kind::kDeviceTokens
                                              : LlmDecodeObservation::Kind::kHeadRows;
      base::Sha256 raw_hash;
      raw_hash.Update(outputs);
      event.raw_output = base::ToHex(raw_hash.Finish());
      if (model.verify_tokens_[slot]) {
        const std::size_t expected = verified.size() * sizeof(std::int32_t);
        if (outputs.size() != expected) {
          observation.incomplete = true;
          return result;
        }
        event.output_tokens.resize(verified.size());
        std::memcpy(event.output_tokens.data(), outputs.data(), expected);
      } else {
        const std::size_t expected = verified.size() * model.logit_width_ * sizeof(float);
        auto rows = model.HeadRows(outputs);
        if (!rows || *rows != verified.size() || outputs.size() < expected) {
          observation.incomplete = true;
          return result;
        }
        const auto* values = reinterpret_cast<const float*>(outputs.data());
        if (!std::ranges::all_of(std::span(values, verified.size() * model.logit_width_),
                                 [](float value) { return std::isfinite(value); }))
          observation.incomplete = true;
        base::Sha256 head_hash;
        head_hash.Update(outputs.first(expected));
        event.head_values = base::ToHex(head_hash.Finish());
      }
      if (result) {
        if (result->tokens.size() > observation.token_limit) {
          observation.incomplete = true;
          return result;
        }
        event.keep = result->keep;
        event.result_tokens.assign(result->tokens.begin(), result->tokens.end());
      }
      return result;
    };
    auto wrapped = engine::DecodingOrchestrator::Make(*model.runner_, *model.composer_,
                                                      std::move(mode), std::move(policy));
    if (!wrapped) return std::unexpected(wrapped.error());
    if (full_head)
      for (auto& drafter : model.drafter_specs_) drafter.device_tokens = false;
    model.speculative_.emplace(std::move(*wrapped));
    return {};
  }
};
}  // namespace llmp::runtime

#endif  // LLMP_TESTS_SUPPORT_LLM_TEST_ACCESS_H_
