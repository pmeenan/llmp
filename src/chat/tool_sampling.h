// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_CHAT_TOOL_SAMPLING_H_
#define LLMP_CHAT_TOOL_SAMPLING_H_
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "chat/tool_calls.h"
#include "tokenizer/tokenizer.h"
namespace llmp::chat {
// Request-local accepted-token cursor. A speculative judge copies it once per
// unit, then advances only accepted candidates. Vocabulary probes borrow the
// pending argument bytes; they never clone argument payloads.
//
// Its grammar is the output parser's own (ToolInterpreter), fed the same
// bytes: each token decoded as the parser's stream decoder decodes it (hidden
// control tokens dropped, ill-formed UTF-8 replaced). In unconstrained auto
// mode nothing is refused per token: only terminals are masked while a call
// is open, and the output parser alone judges the reply. Auto without
// parallel calls is unconstrained until its first call closes (only a second
// call can then be refused).
//
// A step costs only what it restricts: Identity() says when the row is not
// restricted at all (a device's own choice is then exact), Allowed() judges
// one token (a choice it passes is the masked row's choice too), and Mask
// judges every plain token of a state at once (ToolInterpreter::
// PlainVerdict), with one forced-span plan for all of a forced text's
// prefix candidates.
class ToolSampling {
 public:
  // may_reason: the prompt neither opened nor closed the template's
  // reasoning, so the model may open it; a required opener then waits for it.
  // spelling: the template's own whitespace around a forced opener
  // (ToolSpellingFor); empty forces the markers alone.
  ToolSampling(const tokenizer::Tokenizer& tokenizer, ToolCallFormat format, ToolRequest request,
               base::json::Limits limits, bool reasoning, std::span<const std::int32_t> terminals,
               std::span<const std::int32_t> route_stops = {}, bool may_reason = false,
               const ToolOpenerSpelling& spelling = {});
  bool constrained() const;
  // Whether Mask returns the row unchanged now: nothing is restricted, so a
  // device's own greedy choice or verdict is this sampler's.
  bool Identity() const;
  // Mask's verdict on one token now, without masking a row. The masked
  // row's greedy choice is the row's own argmax when Allowed passes it.
  bool Allowed(std::int32_t token) const;
  // The tokens of `ranked` Mask allows, in order, into `out` until it is
  // full (Allowed for each, sharing one step's facts): how many it holds.
  std::size_t AllowedOf(std::span<const std::int32_t> ranked, std::span<std::int32_t> out) const;
  // A request-local refusal (a token outside the grammar, a forced span with
  // no canonical cover, or no allowed candidate), distinct from invalid native
  // logits. A speculative copy that refuses for its request reports it on the
  // request's own cursor (Refuse).
  void Refuse() const { refused_ = true; }
  bool refused() const { return refused_; }
  bool Preserves(std::int32_t token) const;
  bool ReasoningBoundary(std::int32_t token) const;
  bool ContinuesRouteStop(std::int32_t token) const;
  // Masks before temperature/top-k/min-p/top-p; returned storage is caller paid.
  std::span<const float> Mask(std::span<const float> row, std::vector<float>& scratch) const;
  ToolStatus Accept(std::int32_t token);
  // Selected is still unpublished. The returned first token replaces it;
  // remaining tokens become certain proposals in the next common causal unit.
  struct Selection {
    enum class Kind { kOrdinarySelected, kStableSpan } kind = Kind::kOrdinarySelected;
    std::vector<std::int32_t> tokens;
  };
  // OrdinarySelected always contains the unchanged target-selected ID. An open
  // grammar tail remains unconsumed until Accept; it is never a certain span.
  std::expected<Selection, std::string> Plan(std::int32_t selected) const;
  void Budget(std::size_t tokens, std::size_t context) {
    tokens_left_ = tokens;
    context_left_ = context;
  }
  std::span<const std::int32_t> certain() const { return certain_; }
  void Certain(std::vector<std::int32_t> values) { certain_ = std::move(values); }
  void ConsumeCertain(std::size_t count) {
    certain_.erase(certain_.begin(), certain_.begin() + static_cast<std::ptrdiff_t>(count));
  }
  const ToolCallFormat& format() const { return context_->format; }
  // Bytes the grammar cursor buffers (ToolOutputBytes).
  std::size_t buffered() const { return lexical_.buffered(); }

 private:
  struct Context {
    const tokenizer::Tokenizer* tokenizer;
    ToolCallFormat format;
    ToolRequest request;
    std::string prefix;
    std::size_t opener_at = 0;  // where the prefix's opener starts (after its lead)
    // Control-token spans of the prefix's format markers, in prefix bytes.
    // The client's name between them is always encoded as text.
    std::vector<tokenizer::SpecialSpan> specials;
    std::vector<std::int32_t> terminals;
    std::optional<std::string> error;
  };
  // Forcing a required opener: waiting for reasoning to end, choosing between
  // the model's own reasoning opener and the forced prefix, or forcing it.
  struct Trigger {
    bool waiting = false;
    bool choice = false, reasoning_alive = false, prefix_alive = false;
    std::size_t forced = 0;
    std::string tail;
  };
  // What a step's candidates share, found once for all of them.
  struct Step {
    bool constrained = false, can_stop = false, forcing = false;
    std::string_view forced;  // the forced text still owed
    std::vector<tokenizer::SpecialSpan> spans;
    std::optional<std::expected<std::size_t, tokenizer::Error>> cover;
    std::optional<std::expected<tokenizer::ForcedPrefixPlan, tokenizer::Error>> prefixes;
    // The verdict on plain candidates (ToolInterpreter::PlainVerdict), when
    // the trigger cannot tell them apart either.
    std::optional<std::optional<bool>> plain;
    std::string bytes;  // scratch
  };
  Step Facts() const;
  bool Judge(std::int32_t token, Step& step) const;
  bool Plain(std::string_view piece) const;
  bool Transition(std::string_view bytes, Trigger& next) const;
  // The token's own bytes (a hidden control token's are none); false for an
  // ID outside the vocabulary.
  bool Raw(std::int32_t token, std::string_view& out) const;
  bool Lexical(std::int32_t token, tokenizer::StreamDecoder& decoder, std::string& out) const;
  bool TerminalAllowed(std::int32_t token, bool can_stop) const;
  tokenizer::ForcedSpanOptions PlanOptions(std::size_t bytes,
                                           std::span<const tokenizer::SpecialSpan> spans) const;
  std::vector<tokenizer::SpecialSpan> Spans(std::size_t from, std::size_t lead,
                                            std::size_t consumed) const;
  std::expected<Selection, std::string> PlanOnce(std::int32_t selected) const;
  void Extend(Selection& selection) const;
  std::shared_ptr<const Context> context_;
  ToolInterpreter lexical_;
  tokenizer::StreamDecoder decoder_;
  Trigger trigger_;
  bool last_reasoning_boundary_ = false;
  mutable bool refused_ = false;
  std::size_t tokens_left_ = 0, context_left_ = 0;
  std::vector<std::int32_t> certain_;
};
}  // namespace llmp::chat
#endif
