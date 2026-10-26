// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "chat/tool_sampling.h"

#include <algorithm>
#include <limits>
#include <utility>
namespace llmp::chat {
ToolSampling::ToolSampling(const tokenizer::Tokenizer& tokenizer, ToolCallFormat format,
                           ToolRequest request, base::json::Limits limits, bool reasoning,
                           std::span<const std::int32_t> terminals,
                           std::span<const std::int32_t> route_stops)
    : lexical_(format, request, limits) {
  auto context =
      std::make_shared<Context>(Context{.tokenizer = &tokenizer,
                                        .format = std::move(format),
                                        .request = std::move(request),
                                        .prefix = {},
                                        .terminals = {terminals.begin(), terminals.end()}});
  for (const auto token : route_stops) {
    const bool continued =
        token >= 0 && std::cmp_less(token, tokenizer.size()) &&
        std::ranges::find(context->format.continued_route_stops, tokenizer.Text(token)) !=
            context->format.continued_route_stops.end();
    if (!continued) context->terminals.push_back(token);
  }
  const auto trigger = ToolTriggerFor(context->format, context->request);
  if (trigger.required) {
    context->prefix = trigger.begin;
    if (trigger.name) {
      if (!context->format.block_begin.empty()) context->prefix += context->format.call_begin;
      if (context->format.bare_envelope) {
        base::json::AppendQuoted(context->format.name_key, context->prefix);
        context->prefix += ':';
        base::json::AppendQuoted(*trigger.name, context->prefix);
        context->prefix += ',';
      } else
        context->prefix += *trigger.name + context->format.name_end;
    }
  }
  trigger_.waiting = trigger.required && reasoning && !context->format.reasoning_end.empty();
  context_ = std::move(context);
  lexical_.SyntaxOnly();
  lexical_.Reasoning(reasoning);
}
bool ToolSampling::constrained() const {
  const auto kind = context_->request.choice().kind;
  return kind == ToolChoiceKind::kNamed || kind == ToolChoiceKind::kRequired ||
         (kind != ToolChoiceKind::kNone && !context_->request.parallel());
}
bool ToolSampling::Preserves(std::int32_t token) const {
  if (token < 0 || std::cmp_greater_equal(token, context_->tokenizer->size())) return false;
  const auto text = context_->tokenizer->Text(token);
  const auto& f = context_->format;
  for (const auto* marker : {&f.call_begin, &f.call_end, &f.block_begin, &f.block_end,
                             &f.reasoning_end, &f.quote, &f.parameter_begin, &f.parameter_end})
    if (!text.empty() && marker->find(text) != std::string::npos) return true;
  for (const auto& marker : f.call_end_aliases)
    if (!text.empty() && marker.find(text) != std::string::npos) return true;
  for (const auto& header : f.parameter_aliases)
    if (!text.empty() && (header.begin.find(text) != std::string::npos ||
                          header.name_end.find(text) != std::string::npos))
      return true;
  return false;
}
bool ToolSampling::ReasoningBoundary(std::int32_t token) const {
  return (lexical_.reasoning() || last_reasoning_boundary_) && token >= 0 &&
         std::cmp_less(token, context_->tokenizer->size()) &&
         context_->tokenizer->Text(token) == context_->format.reasoning_end;
}
bool ToolSampling::ContinuesRouteStop(std::int32_t token) const {
  return token >= 0 && std::cmp_less(token, context_->tokenizer->size()) &&
         std::ranges::find(context_->format.continued_route_stops,
                           context_->tokenizer->Text(token)) !=
             context_->format.continued_route_stops.end();
}
bool ToolSampling::TerminalAllowed(std::int32_t token, bool can_stop) const {
  if ((!trigger_.waiting && can_stop) || ReasoningBoundary(token)) return true;
  return std::ranges::find(context_->terminals, token) == context_->terminals.end();
}
bool ToolSampling::Decode(std::int32_t token, std::string& out) const {
  if (token < 0 || std::cmp_greater_equal(token, context_->tokenizer->size())) return false;
  if (context_->tokenizer->Kind(token) == tokenizer::TokenKind::kControl && !Preserves(token))
    return true;
  return context_->tokenizer->Decode(std::span(&token, 1), {.control_tokens = true}, out)
      .has_value();
}
bool ToolSampling::Transition(std::string_view bytes, Trigger& next) const {
  if (context_->prefix.empty()) return true;
  const auto& f = context_->format;
  const auto opener =
      f.block_begin.empty() ? std::string_view(f.call_begin) : std::string_view(f.block_begin);
  const auto held =
      std::max(f.opener_ends_reasoning ? opener.size() : std::size_t{0}, f.reasoning_end.size());
  for (const char byte : bytes) {
    if (next.waiting) {
      next.tail += byte;
      if (f.opener_ends_reasoning && !opener.empty() && next.tail.ends_with(opener)) {
        next.waiting = false;
        next.forced = opener.size();
        next.tail.clear();
      } else if (!f.reasoning_end.empty() && next.tail.ends_with(f.reasoning_end)) {
        next.waiting = false;
        next.forced = 0;
        next.tail.clear();
      } else if (next.tail.size() > held) {
        next.tail.erase(0, next.tail.size() - held);
      }
    } else if (next.forced < context_->prefix.size()) {
      if (byte != context_->prefix[next.forced]) return false;
      ++next.forced;
    }
  }
  return true;
}
std::expected<ToolSampling::Selection, std::string> ToolSampling::Plan(
    std::int32_t selected) const {
  if (!TerminalAllowed(selected, lexical_.CanStop()))
    return std::unexpected("selected terminal would leave an incomplete tool call");
  std::string piece;
  auto next = trigger_;
  if (!Decode(selected, piece) || !Transition(piece, next) || !lexical_.Allows(piece))
    return std::unexpected("selected token violates the tool constraint");
  if (next.waiting || next.forced >= context_->prefix.size())
    return Selection{.tokens = {selected}};
  const auto forced = std::string_view(context_->prefix).substr(next.forced);
  tokenizer::ForcedSpanOptions options;
  options.encode.max_bytes = piece.size() + forced.size();
  options.encode.max_tokens = tokens_left_;
  options.remaining_tokens = tokens_left_;
  options.remaining_context = context_left_;
  auto planned = context_->tokenizer->PlanForcedSpan(selected, forced, options);
  if (!planned) return std::unexpected(planned.error().ToString());
  if (!planned->selected_covered || planned->tokens.empty()) {
    // The final open block remains ordinary target sampling. This is a local
    // consistency check, not canonicality under unknown future raw-BPE merges.
    // No replacement or certain continuation from that block is injected.
    if (planned->selected_matches_canonical_prefix) return Selection{.tokens = {selected}};
    return std::unexpected("tool boundary has no publishable canonical cover");
  }
  return Selection{.kind = Selection::Kind::kStableSpan, .tokens = std::move(planned->tokens)};
}
std::span<const float> ToolSampling::Mask(std::span<const float> row,
                                          std::vector<float>& scratch) const {
  const bool constrained = this->constrained();
  const bool can_stop = lexical_.CanStop();
  if (!constrained && can_stop && !trigger_.waiting) return row;
  scratch.assign(row.begin(), row.end());
  const bool forcing = !trigger_.waiting && trigger_.forced < context_->prefix.size();
  std::expected<std::size_t, tokenizer::Error> cover = std::size_t{0};
  if (forcing) {
    tokenizer::ForcedSpanOptions cover_options;
    const auto forced = std::string_view(context_->prefix).substr(trigger_.forced);
    cover_options.encode.max_bytes = forced.size();
    cover = context_->tokenizer->ForcedCoverBytes(forced, cover_options);
  }
  std::string piece;
  for (std::size_t i = 0; i < row.size(); ++i) {
    const auto token = static_cast<std::int32_t>(i);
    bool allowed = TerminalAllowed(token, can_stop);
    if (!constrained) {
      if (!allowed) scratch[i] = -std::numeric_limits<float>::infinity();
      continue;
    }
    if (allowed && forcing) {
      allowed = cover && context_->tokenizer->CoversForced(token, *cover);
    }
    if (allowed) {
      piece.clear();
      auto next = trigger_;
      allowed = Decode(token, piece) && Transition(piece, next) && lexical_.Allows(piece);
      // A hidden control cannot advance a forced prefix. Otherwise it could
      // terminate without ever producing the required opening sequence.
      if (allowed && forcing && piece.empty()) allowed = false;
      if (allowed && !next.waiting && next.forced < context_->prefix.size())
        allowed = Plan(token).has_value();
    }
    if (!allowed) scratch[i] = -std::numeric_limits<float>::infinity();
  }
  return scratch;
}
ToolStatus ToolSampling::Accept(std::int32_t token) {
  if (!TerminalAllowed(token, lexical_.CanStop()))
    return std::unexpected("accepted terminal would leave an incomplete tool call");
  std::string piece;
  if (!Decode(token, piece) || !Transition(piece, trigger_) || !lexical_.Allows(piece))
    return std::unexpected("accepted token violates the tool constraint");
  const bool reasoning = lexical_.reasoning();
  auto result = lexical_.Push(piece, [](ToolEvent) { return true; });
  last_reasoning_boundary_ = reasoning && !lexical_.reasoning();
  if (result) {
    tokens_left_ -= tokens_left_ != 0 ? 1 : 0;
    context_left_ -= context_left_ != 0 ? 1 : 0;
  }
  return result;
}
}  // namespace llmp::chat
