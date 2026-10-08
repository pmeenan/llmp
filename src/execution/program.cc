// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "execution/program.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "catalog/catalog.h"
#include "memory/commitment.h"
#include "model/context.h"
#include "model/state.h"

namespace llmp::execution {
namespace {

std::unexpected<ProgramRejection> Reject(ProgramError error, std::string detail,
                                         std::string phase = {}, std::uint64_t width = 0) {
  return std::unexpected(ProgramRejection{.error = error,
                                          .phase = std::move(phase),
                                          .width = width,
                                          .required = {},
                                          .shortfall = {},
                                          .detail = std::move(detail)});
}

std::optional<std::uint64_t> Multiply(std::uint64_t a, std::uint64_t b) {
  std::uint64_t product = 0;
  if (__builtin_mul_overflow(a, b, &product)) {
    return std::nullopt;
  }
  return product;
}

std::uint64_t CeilDiv(std::uint64_t a, std::uint64_t b) { return (a / b) + (a % b != 0 ? 1 : 0); }

// Whole backing units of `granularity`.
std::optional<Bytes> RoundUp(Bytes bytes, Bytes granularity) {
  return Multiply(CeilDiv(bytes.value(), granularity.value()), granularity.value())
      .transform([](std::uint64_t value) { return Bytes(value); });
}

bool Whole(Bytes bytes, Bytes granularity) { return bytes.value() % granularity.value() == 0; }

// The width a phase kind's rule fixes; kChunk chooses at planning.
std::uint64_t FixedWidth(WidthRule rule, const DecodingMode& mode, const ProgramRequest& request) {
  switch (rule) {
    case WidthRule::kChunk:
    case WidthRule::kOne:
      return 1;
    case WidthRule::kDraft:
      return request.draft_depth;
    case WidthRule::kVerify:
      return std::uint64_t{request.draft_depth} + 1;
    case WidthRule::kBlock:
      return mode.block;
  }
  return 1;
}

// A phase kind being planned.
struct Candidate {
  const PhaseKind* kind = nullptr;
  std::vector<std::uint64_t> allowed;  // widths it may run at, once chosen
  Bytes closure;
};

// The largest rounded working set among `widths` of `kind`.
std::optional<Bytes> WorkingAt(const PhaseKind& kind, const std::vector<std::uint64_t>& widths,
                               Bytes granularity) {
  Bytes largest;
  for (const PhaseWidth& width : kind.widths) {
    if (std::ranges::find(widths, width.positions) == widths.end()) {
      continue;
    }
    const std::optional<Bytes> rounded = RoundUp(width.working, granularity);
    if (!rounded) {
      return std::nullopt;
    }
    largest = std::max(largest, *rounded);
  }
  return largest;
}

std::expected<void, ProgramRejection> CheckMode(const DecodingMode& mode,
                                                const ProgramRequest& request) {
  const bool speculative = mode.max_draft != 0;
  const bool block = mode.block != 0;
  OutputUnit unit = OutputUnit::kToken;
  if (speculative) {
    unit = OutputUnit::kAcceptedRun;
  } else if (block) {
    unit = OutputUnit::kBlock;
  }
  if ((speculative && block) || mode.unit != unit || (block && mode.max_steps == 0)) {
    return Reject(ProgramError::kInvalid,
                  std::format("decoding mode {} is not one consistent mode", mode.name));
  }
  if (request.prompt == 0 || request.max_output == 0) {
    return Reject(ProgramError::kInvalid, "a request needs a prompt and an output bound");
  }
  if (speculative ? (request.draft_depth == 0 || request.draft_depth > mode.max_draft)
                  : request.draft_depth != 0) {
    return Reject(ProgramError::kUnsupported, std::format("draft depth {} under decoding mode {}",
                                                          request.draft_depth, mode.name));
  }
  if (block ? (request.steps == 0 || request.steps > mode.max_steps) : request.steps != 0) {
    return Reject(
        ProgramError::kUnsupported,
        std::format("{} denoising steps under decoding mode {}", request.steps, mode.name));
  }
  return {};
}

}  // namespace

std::string ToString(ProgramError error) {
  switch (error) {
    case ProgramError::kInvalid:
      return "invalid program contract or request";
    case ProgramError::kUnsupported:
      return "not validated in this contract";
    case ProgramError::kContextExhausted:
      return "the context has no room for the output";
    case ProgramError::kDoesNotFit:
      return "a phase envelope does not fit above the retained state";
    case ProgramError::kOverflow:
      return "the envelope's arithmetic would overflow";
    case ProgramError::kCatalog:
      return "the context's resources are not in the catalog";
    case ProgramError::kBeyondProgram:
      return "the phase is past the admitted program";
    case ProgramError::kWrongState:
      return "the phase is not in the state this needs";
    case ProgramError::kOutputFull:
      return "the output buffer lacks room for the phase's output";
  }
  return "unknown program error";
}

std::string ToString(const ProgramRejection& rejection) {
  std::string text = ToString(rejection.error);
  if (!rejection.phase.empty()) {
    text += std::format(": phase {} at width {}", rejection.phase, rejection.width);
  }
  if (rejection.required != Bytes()) {
    text += std::format(" needs {} bytes, {} more than available", rejection.required.value(),
                        rejection.shortfall.value());
  }
  if (!rejection.detail.empty()) {
    text += std::format(" ({})", rejection.detail);
  }
  return text;
}

const PlannedPhase* ProgramPlan::Phase(std::string_view kind) const {
  const auto found = std::ranges::find(phases, kind, &PlannedPhase::kind);
  return found != phases.end() ? &*found : nullptr;
}

const RetainedItem* ProgramPlan::Retained(std::string_view name) const {
  const auto found = std::ranges::find(retained, name, &RetainedItem::name);
  return found != retained.end() ? &*found : nullptr;
}

std::expected<ProgramPlan, ProgramRejection> PlanProgram(const ProgramContract& contract,
                                                         const model::ModelContext& context,
                                                         const catalog::Catalog& catalog,
                                                         catalog::DomainId domain,
                                                         const ProgramRequest& request,
                                                         Bytes available, Bytes granularity) {
  const DecodingMode& mode = contract.mode;
  if (granularity == Bytes()) {
    return Reject(ProgramError::kInvalid, "no backing granularity");
  }
  if (auto checked = CheckMode(mode, request); !checked) {
    return std::unexpected(checked.error());
  }
  const bool speculative = mode.max_draft != 0;
  const bool block = mode.block != 0;

  // The output bound, then each live state's bound.
  ProgramPlan plan;
  plan.prompt = request.prompt;
  if (request.prompt >= contract.context_limit) {
    return Reject(ProgramError::kContextExhausted, "the prompt fills the context");
  }
  const std::uint64_t room = contract.context_limit - request.prompt;
  if (block) {
    // End on a block the context holds; commits clamp to the output bound.
    const std::uint64_t whole = room / mode.block;
    if (whole == 0) {
      return Reject(ProgramError::kContextExhausted, "no whole block fits after the prompt");
    }
    plan.blocks = std::min(whole, CeilDiv(request.max_output, mode.block));
    plan.output = std::min(request.max_output, plan.blocks * mode.block);
  } else {
    if (request.max_output > room) {
      return Reject(ProgramError::kContextExhausted, "the output bound exceeds the context");
    }
    plan.output = request.max_output;
  }
  plan.state_bound = request.prompt + plan.output;  // at most the context limit

  // R_i: the live states, the working states, then the output buffer,
  // each under a name of its own, so no item's allowance can be read for
  // another's.
  std::set<std::string_view> retained_names = {kOutputName};
  const bool appends = std::ranges::any_of(contract.phases, &PhaseKind::appends);
  for (const model::StateRepresentation& state : contract.states) {
    if (state.name.empty() || !retained_names.insert(state.name).second) {
      return Reject(ProgramError::kInvalid,
                    std::format("state {} is not uniquely named", state.name));
    }
    if (!model::IsValid(state) || !Whole(state.block_bytes, granularity) ||
        !Whole(state.snapshot_bytes, granularity)) {
      return Reject(ProgramError::kInvalid,
                    std::format("state {} is not whole backing blocks", state.name));
    }
    if (appends && !state.Can(model::StateCapability::kAppend)) {
      return Reject(ProgramError::kUnsupported, std::format("state {} cannot append", state.name));
    }
    // A rejected draft needs truncation, to any position or through a
    // snapshot at the prefix and after every drafted position: one more
    // than the draft depth.
    if (speculative && (!state.CanTruncate() || (!state.Can(model::StateCapability::kTruncate) &&
                                                 state.max_snapshots <= request.draft_depth))) {
      return Reject(ProgramError::kUnsupported,
                    std::format("speculative decoding needs state {} to truncate after a "
                                "rejected draft of {}",
                                state.name, request.draft_depth));
    }
    const std::optional<Bytes> allowance = model::StateAllowance(state, plan.state_bound);
    if (!allowance) {
      return Reject(ProgramError::kOverflow, std::format("state {}", state.name));
    }
    plan.retained.push_back({.name = state.name, .bytes = *allowance});
  }
  for (const WorkingState& working : contract.working) {
    const std::optional<Bytes> bytes = RoundUp(working.bytes, granularity);
    if (working.name.empty() || !retained_names.insert(working.name).second || !bytes) {
      return Reject(ProgramError::kInvalid, std::format("working state {}", working.name));
    }
    plan.retained.push_back({.name = working.name, .bytes = *bytes});
  }

  // Each phase kind's widths and closure. Some phase must emit, in units
  // of some wire bytes, or the output buffer would be empty and no phase
  // could reserve room in it: refused here, not once running.
  if (contract.phases.empty()) {
    return Reject(ProgramError::kInvalid, "a contract needs a phase kind");
  }
  if (!std::ranges::any_of(contract.phases, &PhaseKind::emits) ||
      mode.wire_per_position == Bytes()) {
    return Reject(ProgramError::kInvalid,
                  std::format("no phase kind emits output under decoding mode {}", mode.name));
  }
  std::set<std::string_view> names;
  std::vector<Candidate> candidates;
  Bytes output_capacity;
  for (const PhaseKind& kind : contract.phases) {
    const auto invalid = [&](std::string detail) {
      return Reject(ProgramError::kInvalid, std::move(detail), kind.name);
    };
    if (kind.name.empty() || !names.insert(kind.name).second) {
      return invalid("a phase kind needs a unique name");
    }
    // Sorted under <= means no width repeats or descends: strictly ascending.
    if (kind.widths.empty() || kind.widths.front().positions == 0 ||
        !std::ranges::is_sorted(kind.widths, std::ranges::less_equal{}, &PhaseWidth::positions)) {
      return invalid("validated widths must be positive and strictly ascending");
    }
    for (const model::ComponentRole role : kind.components) {
      if (std::ranges::find(context.components(), role, &model::Component::role) ==
          context.components().end()) {
        return invalid(std::format("the context has no {} component", model::ToString(role)));
      }
    }
    const bool needs_block = kind.rule == WidthRule::kBlock || kind.repeat == Repeat::kBlockSteps ||
                             kind.repeat == Repeat::kBlocks;
    const bool needs_draft = kind.rule == WidthRule::kDraft || kind.rule == WidthRule::kVerify;
    if ((needs_block && !block) || (needs_draft && !speculative) ||
        (kind.rule == WidthRule::kChunk && kind.emits)) {
      return invalid(
          std::format("its width or repeat rule does not suit decoding mode {}", mode.name));
    }
    Candidate candidate{.kind = &kind, .allowed = {}, .closure = {}};
    for (const PhaseWidth& width : kind.widths) {
      candidate.allowed.push_back(width.positions);
    }
    if (kind.rule != WidthRule::kChunk) {
      const std::uint64_t width = FixedWidth(kind.rule, mode, request);
      if (std::ranges::find(candidate.allowed, width) == candidate.allowed.end()) {
        return Reject(ProgramError::kUnsupported, "no validated plan at this width", kind.name,
                      width);
      }
      if (kind.rule == WidthRule::kBlock) {
        candidate.allowed = {width};  // a canvas keeps the model's block
      } else {
        std::erase_if(candidate.allowed, [&](std::uint64_t w) { return w > width; });
      }
      // Near the bound a draft or verify narrows: every narrower width
      // must be validated too.
      if (needs_draft) {
        for (std::uint64_t w = 1; w <= width; ++w) {
          if (std::ranges::find(candidate.allowed, w) == candidate.allowed.end()) {
            return Reject(ProgramError::kUnsupported,
                          "a clamped draft or verify needs every narrower width", kind.name, w);
          }
        }
      }
      if (kind.emits) {
        const std::optional<std::uint64_t> bytes = Multiply(width, mode.wire_per_position.value());
        if (!bytes) {
          return Reject(ProgramError::kOverflow, "the output buffer", kind.name, width);
        }
        output_capacity = std::max(output_capacity, Bytes(*bytes));
      }
    }
    auto closure = context.ClosureOf(catalog, kind.components);
    if (!closure) {
      return Reject(ProgramError::kCatalog, catalog::ToString(closure.error()), kind.name);
    }
    // A closure is planned in one domain; bytes in another would be in no
    // envelope at all.
    if (std::ranges::any_of(closure->bytes_by_domain, [&](const auto& entry) {
          return entry.first != domain && entry.second != Bytes();
        })) {
      return Reject(ProgramError::kUnsupported, "its closure reaches another memory domain",
                    kind.name);
    }
    const auto in_domain = closure->bytes_by_domain.find(domain);
    candidate.closure = in_domain != closure->bytes_by_domain.end() ? in_domain->second : Bytes();
    candidates.push_back(std::move(candidate));
  }
  const std::optional<Bytes> output_buffer = RoundUp(output_capacity, granularity);
  if (!output_buffer) {
    return Reject(ProgramError::kOverflow, "the output buffer");
  }
  plan.output_capacity = output_capacity;
  plan.retained.push_back({.name = std::string(kOutputName), .bytes = *output_buffer});
  Bytes retained;
  for (const RetainedItem& item : plan.retained) {
    const std::optional<Bytes> sum = retained.Plus(item.bytes);
    if (!sum) {
      return Reject(ProgramError::kOverflow, std::format("retained state {}", item.name));
    }
    retained = *sum;
  }
  plan.envelope.retained = retained;

  // E_i: each phase's envelope above R_i, at the widest width it may run
  // at; a chunked kind takes its widest width that fits.
  for (Candidate& candidate : candidates) {
    const PhaseKind& kind = *candidate.kind;
    std::optional<ProgramRejection> narrowest;
    std::optional<PlannedPhase> planned;
    // A chunked kind tries its widths from the widest down.
    const std::size_t tries = kind.rule == WidthRule::kChunk ? candidate.allowed.size() : 1;
    for (std::size_t i = 0; i < tries && !planned; ++i) {
      std::vector<std::uint64_t> allowed = candidate.allowed;
      allowed.resize(candidate.allowed.size() - i);
      const std::optional<Bytes> working = WorkingAt(kind, allowed, granularity);
      const std::optional<Bytes> envelope =
          working.and_then([&](Bytes w) { return candidate.closure.Plus(w); });
      const std::optional<Bytes> required =
          envelope.and_then([&](Bytes e) { return retained.Plus(e); });
      if (!required) {
        return Reject(ProgramError::kOverflow, "the phase envelope", kind.name, allowed.back());
      }
      if (*required > available) {
        narrowest = ProgramRejection{.error = ProgramError::kDoesNotFit,
                                     .phase = kind.name,
                                     .width = allowed.back(),
                                     .required = *required,
                                     .shortfall = required->Minus(available).value_or(Bytes()),
                                     .detail = {}};
        continue;
      }
      std::optional<std::uint64_t> count;
      switch (kind.repeat) {
        case Repeat::kPromptChunks:
          count = CeilDiv(request.prompt, allowed.front());
          break;
        case Repeat::kOutputSteps:
          count = plan.output;
          break;
        case Repeat::kBlockSteps:
          count = Multiply(plan.blocks, request.steps);
          break;
        case Repeat::kBlocks:
          count = plan.blocks;
          break;
      }
      if (!count) {
        return Reject(ProgramError::kOverflow, "the phase count", kind.name, allowed.back());
      }
      planned = PlannedPhase{.kind = kind.name,
                             .width = allowed.back(),
                             .widths = allowed,
                             .closure = candidate.closure,
                             .working = *working,
                             .envelope = *envelope,
                             .max_count = *count};
    }
    if (!planned) {
      return std::unexpected(*narrowest);
    }
    plan.envelope.phase = std::max(plan.envelope.phase, planned->envelope);
    const std::optional<std::uint64_t> phases =
        plan.max_phases > UINT64_MAX - planned->max_count
            ? std::nullopt
            : std::optional<std::uint64_t>(plan.max_phases + planned->max_count);
    if (!phases) {
      return Reject(ProgramError::kOverflow, "the phase count", kind.name);
    }
    plan.max_phases = *phases;
    plan.phases.push_back(*std::move(planned));
  }
  return plan;
}

ProgramCursor::ProgramCursor(const ProgramPlan& plan) {
  for (const PlannedPhase& phase : plan.phases) {
    kinds_.emplace(phase.kind, Kind{.widths = phase.widths, .max_count = phase.max_count});
  }
}

std::expected<void, ProgramError> ProgramCursor::Begin(std::string_view kind, std::uint64_t width) {
  if (!current_.empty()) {
    return std::unexpected(ProgramError::kWrongState);  // one phase at a time
  }
  const auto found = kinds_.find(kind);
  if (found == kinds_.end()) {
    return std::unexpected(ProgramError::kInvalid);
  }
  Kind& planned = found->second;
  if (std::ranges::find(planned.widths, width) == planned.widths.end()) {
    return std::unexpected(ProgramError::kUnsupported);
  }
  if (planned.count >= planned.max_count) {
    return std::unexpected(ProgramError::kBeyondProgram);  // outcomes only shorten it
  }
  ++planned.count;
  ++phases_run_;
  current_ = found->first;
  return {};
}

std::expected<void, ProgramError> ProgramCursor::End() {
  if (current_.empty()) {
    return std::unexpected(ProgramError::kWrongState);
  }
  current_.clear();
  return {};
}

std::uint64_t ProgramCursor::CountOf(std::string_view kind) const {
  const auto found = kinds_.find(kind);
  return found != kinds_.end() ? found->second.count : 0;
}

std::expected<void, ProgramError> OutputBuffer::Reserve(Bytes bytes, std::uint64_t positions) {
  if (reserved_ != Bytes()) {
    return std::unexpected(ProgramError::kWrongState);  // one phase at a time
  }
  if (bytes == Bytes() || positions == 0) {
    return std::unexpected(ProgramError::kInvalid);
  }
  if (positions > bound_ - published_positions_) {
    return std::unexpected(ProgramError::kBeyondProgram);  // past the admitted output bound
  }
  const std::optional<Bytes> after = used_.Plus(bytes);
  if (!after || *after > capacity_) {
    return std::unexpected(ProgramError::kOutputFull);
  }
  reserved_ = bytes;
  reserved_positions_ = positions;
  return {};
}

std::expected<void, ProgramError> OutputBuffer::Publish(Bytes bytes, std::uint64_t positions) {
  if (reserved_ == Bytes()) {
    return std::unexpected(ProgramError::kWrongState);
  }
  if (bytes > reserved_ || positions > reserved_positions_) {
    return std::unexpected(ProgramError::kInvalid);
  }
  used_ = used_.Plus(bytes).value_or(capacity_);  // within the reservation, so within capacity
  published_positions_ += positions;              // within the reservation, so within the bound
  Cancel();
  return {};
}

std::expected<void, ProgramError> OutputBuffer::Drain(Bytes bytes) {
  const std::optional<Bytes> left = used_.Minus(bytes);
  if (!left) {
    return std::unexpected(ProgramError::kInvalid);
  }
  used_ = *left;
  return {};
}

}  // namespace llmp::execution
