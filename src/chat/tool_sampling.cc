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
                           std::span<const std::int32_t> route_stops, bool may_reason)
    : lexical_(format, request, limits), decoder_(tokenizer, {.control_tokens = true}) {
  auto context =
      std::make_shared<Context>(Context{.tokenizer = &tokenizer,
                                        .format = std::move(format),
                                        .request = std::move(request),
                                        .prefix = {},
                                        .specials = {},
                                        .terminals = {terminals.begin(), terminals.end()},
                                        .error = std::nullopt});
  for (const auto token : route_stops) {
    const bool continued =
        token >= 0 && std::cmp_less(token, tokenizer.size()) &&
        std::ranges::find(context->format.continued_route_stops, tokenizer.Text(token)) !=
            context->format.continued_route_stops.end();
    if (!continued) context->terminals.push_back(token);
  }
  // Format markers are encoded with their control tokens; the client's name
  // is always text, whatever control-token text it contains (marked spans).
  const auto marker = [&](std::string_view text) {
    if (auto spans = tokenizer.ControlSpans(text)) {
      for (auto span : *spans) {
        span.offset += context->prefix.size();
        context->specials.push_back(span);
      }
    } else {
      context->error = spans.error().ToString();
    }
    context->prefix += text;
  };
  const auto trigger = ToolTriggerFor(context->format, context->request);
  if (trigger.required) {
    marker(trigger.begin);
    if (trigger.name) {
      if (!context->format.block_begin.empty()) marker(context->format.call_begin);
      if (context->format.bare_envelope) {
        std::string key;
        base::json::AppendQuoted(context->format.name_key, key);
        marker(key + ":");
        base::json::AppendQuoted(*trigger.name, context->prefix);
        marker(",");
      } else {
        context->prefix += *trigger.name;
        marker(context->format.name_end);
      }
    }
  }
  const auto& f = context->format;
  trigger_.waiting = trigger.required && reasoning && !f.reasoning_end.empty();
  trigger_.choice = trigger.required && !reasoning && may_reason && !f.reasoning_begin.empty() &&
                    !f.reasoning_end.empty();
  trigger_.reasoning_alive = trigger_.prefix_alive = trigger_.choice;
  context_ = std::move(context);
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
  for (const auto* marker :
       {&f.call_begin, &f.call_end, &f.block_begin, &f.block_end, &f.reasoning_end,
        &f.reasoning_begin, &f.quote, &f.parameter_begin, &f.parameter_end})
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
  if ((!trigger_.waiting && !trigger_.choice && can_stop) || ReasoningBoundary(token)) return true;
  return std::ranges::find(context_->terminals, token) == context_->terminals.end();
}
// The token's own bytes, for matching the forced prefix and planning.
bool ToolSampling::Raw(std::int32_t token, std::string& out) const {
  if (token < 0 || std::cmp_greater_equal(token, context_->tokenizer->size())) return false;
  if (context_->tokenizer->Kind(token) == tokenizer::TokenKind::kControl && !Preserves(token))
    return true;
  return context_->tokenizer->Decode(std::span(&token, 1), {.control_tokens = true}, out)
      .has_value();
}
// The bytes the output parser receives for this token (ChatOutput's decoder).
bool ToolSampling::Lexical(std::int32_t token, tokenizer::StreamDecoder& decoder,
                           std::string& out) const {
  if (token < 0 || std::cmp_greater_equal(token, context_->tokenizer->size())) return false;
  if (context_->tokenizer->Kind(token) == tokenizer::TokenKind::kControl && !Preserves(token))
    return true;
  return decoder.Push(token, out).has_value();
}
bool ToolSampling::Transition(std::string_view bytes, Trigger& next) const {
  if (context_->prefix.empty()) return true;
  const auto& f = context_->format;
  const auto& prefix = context_->prefix;
  const auto opener =
      f.block_begin.empty() ? std::string_view(f.call_begin) : std::string_view(f.block_begin);
  const auto held =
      std::max(f.opener_ends_reasoning ? opener.size() : std::size_t{0}, f.reasoning_end.size());
  for (const char byte : bytes) {
    if (next.choice) {
      // Either the model's own reasoning opener or the forced prefix.
      const auto at = next.forced;
      const bool reasoning =
          next.reasoning_alive && at < f.reasoning_begin.size() && f.reasoning_begin[at] == byte;
      const bool forced = next.prefix_alive && at < prefix.size() && prefix[at] == byte;
      if (!reasoning && !forced) return false;
      ++next.forced;
      next.reasoning_alive = reasoning;
      next.prefix_alive = forced;
      if (reasoning && next.forced == f.reasoning_begin.size()) {
        next.choice = false;
        next.waiting = true;
        next.forced = 0;
        next.tail.clear();
      } else if (!reasoning) {
        next.choice = false;  // forcing the rest of the prefix
      }
    } else if (next.waiting) {
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
    } else if (next.forced < prefix.size()) {
      if (byte != prefix[next.forced]) return false;
      ++next.forced;
    }
  }
  return true;
}
// The prefix's control spans from prefix byte `from` on, in planned-text
// coordinates: `lead` bytes precede it, of which the last `consumed` are the
// prefix bytes just before `from` (a span may begin inside the selected token).
std::vector<tokenizer::SpecialSpan> ToolSampling::Spans(std::size_t from, std::size_t lead,
                                                        std::size_t consumed) const {
  std::vector<tokenizer::SpecialSpan> out;
  const auto first = from - std::min(from, consumed);
  for (const auto& span : context_->specials) {
    if (span.offset < first) continue;
    out.push_back({.offset = span.offset - from + lead, .length = span.length});
  }
  return out;
}
std::expected<ToolSampling::Selection, std::string> ToolSampling::PlanOnce(
    std::int32_t selected) const {
  if (!constrained()) return Selection{.tokens = {selected}};
  if (context_->error) return std::unexpected(*context_->error);
  if (!TerminalAllowed(selected, lexical_.CanStop()))
    return std::unexpected("selected terminal would leave an incomplete tool call");
  std::string piece, bytes;
  auto decoder = decoder_;
  auto next = trigger_;
  if (!Raw(selected, piece) || !Lexical(selected, decoder, bytes) || !Transition(piece, next) ||
      !lexical_.Allows(bytes))
    return std::unexpected("selected token violates the tool constraint");
  if (next.waiting || next.choice || next.forced >= context_->prefix.size())
    return Selection{.tokens = {selected}};
  const auto forced = std::string_view(context_->prefix).substr(next.forced);
  const auto start = trigger_.waiting ? std::size_t{0} : trigger_.forced;
  const auto spans = Spans(next.forced, piece.size(), std::min(piece.size(), next.forced - start));
  tokenizer::ForcedSpanOptions options;
  options.encode.special = tokenizer::SpecialTokens::kUserDefinedOnly;
  options.encode.max_bytes = piece.size() + forced.size();
  options.encode.max_tokens = tokens_left_;
  options.marked = true;
  options.specials = spans;
  options.remaining_tokens = tokens_left_;
  options.remaining_context = context_left_;
  auto planned = context_->tokenizer->PlanForcedSpan(selected, forced, options);
  if (!planned) return std::unexpected(planned.error().ToString());
  if (!planned->selected_covered || planned->tokens.empty()) {
    // The final open block remains ordinary target sampling. This is a local
    // consistency check, not canonicality under unknown future raw-BPE merges.
    // No replacement or certain continuation from that block is injected. A
    // plan cut by the remaining allowance ends the reply by length instead.
    if (planned->selected_matches_canonical_prefix || planned->limited)
      return Selection{.tokens = {selected}};
    return std::unexpected("tool boundary has no publishable canonical cover");
  }
  return Selection{.kind = Selection::Kind::kStableSpan, .tokens = std::move(planned->tokens)};
}
std::expected<ToolSampling::Selection, std::string> ToolSampling::Plan(
    std::int32_t selected) const {
  auto planned = PlanOnce(selected);
  if (!planned) Refuse();
  return planned;
}
std::span<const float> ToolSampling::Mask(std::span<const float> row,
                                          std::vector<float>& scratch) const {
  const bool constrained = this->constrained();
  const bool can_stop = lexical_.CanStop();
  if (!constrained && can_stop && !trigger_.waiting && !trigger_.choice) return row;
  scratch.assign(row.begin(), row.end());
  const bool forcing =
      !trigger_.waiting && !trigger_.choice && trigger_.forced < context_->prefix.size();
  std::expected<std::size_t, tokenizer::Error> cover = std::size_t{0};
  std::vector<tokenizer::SpecialSpan> cover_spans;
  if (forcing) {
    tokenizer::ForcedSpanOptions cover_options;
    const auto forced = std::string_view(context_->prefix).substr(trigger_.forced);
    cover_spans = Spans(trigger_.forced, 0, 0);
    cover_options.encode.special = tokenizer::SpecialTokens::kUserDefinedOnly;
    cover_options.encode.max_bytes = forced.size();
    cover_options.marked = true;
    cover_options.specials = cover_spans;
    cover = context_->tokenizer->ForcedCoverBytes(forced, cover_options);
  }
  std::string piece, bytes;
  for (std::size_t i = 0; i < row.size(); ++i) {
    const auto token = static_cast<std::int32_t>(i);
    bool allowed = TerminalAllowed(token, can_stop);
    if (!constrained) {
      if (!allowed) scratch[i] = -std::numeric_limits<float>::infinity();
      continue;
    }
    if (allowed && forcing) allowed = cover && context_->tokenizer->CoversForced(token, *cover);
    if (allowed) {
      piece.clear();
      bytes.clear();
      auto decoder = decoder_;
      auto next = trigger_;
      allowed = Raw(token, piece) && Lexical(token, decoder, bytes) && Transition(piece, next) &&
                lexical_.Allows(bytes);
      // A hidden control cannot advance a forced prefix (or the choice of
      // one). Otherwise it could end without ever producing the opener.
      if (allowed && (forcing || trigger_.choice) && piece.empty()) allowed = false;
      if (allowed && !next.waiting && !next.choice && next.forced < context_->prefix.size())
        allowed = PlanOnce(token).has_value();
    }
    if (!allowed) scratch[i] = -std::numeric_limits<float>::infinity();
  }
  return scratch;
}
ToolStatus ToolSampling::Accept(std::int32_t token) {
  std::string piece, bytes;
  if (!constrained()) {
    // Unconstrained auto: nothing is refused per token. The cursor follows the
    // output so terminals stay masked inside an open call; a reply outside
    // the grammar is the output parser's to report.
    if (!Lexical(token, decoder_, bytes)) {
      Refuse();
      return std::unexpected("accepted token is outside the vocabulary");
    }
    const bool reasoning = lexical_.reasoning();
    (void)lexical_.Push(bytes, [](ToolEvent) { return true; });
    last_reasoning_boundary_ = reasoning && !lexical_.reasoning();
  } else {
    auto decoder = decoder_;
    if (!TerminalAllowed(token, lexical_.CanStop())) {
      Refuse();
      return std::unexpected("accepted terminal would leave an incomplete tool call");
    }
    if (!Raw(token, piece) || !Lexical(token, decoder, bytes) || !Transition(piece, trigger_) ||
        !lexical_.Allows(bytes)) {
      Refuse();
      return std::unexpected("accepted token violates the tool constraint");
    }
    decoder_ = std::move(decoder);
    const bool reasoning = lexical_.reasoning();
    if (auto pushed = lexical_.Push(bytes, [](ToolEvent) { return true; }); !pushed) {
      Refuse();
      return pushed;
    }
    last_reasoning_boundary_ = reasoning && !lexical_.reasoning();
  }
  tokens_left_ -= tokens_left_ != 0 ? 1 : 0;
  context_left_ -= context_left_ != 0 ? 1 : 0;
  return {};
}
}  // namespace llmp::chat
