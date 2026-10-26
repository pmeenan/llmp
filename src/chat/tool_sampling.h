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
// is open, and the output parser alone judges the reply.
class ToolSampling {
 public:
  // may_reason: the prompt neither opened nor closed the template's
  // reasoning, so the model may open it; a required opener then waits for it.
  ToolSampling(const tokenizer::Tokenizer& tokenizer, ToolCallFormat format, ToolRequest request,
               base::json::Limits limits, bool reasoning, std::span<const std::int32_t> terminals,
               std::span<const std::int32_t> route_stops = {}, bool may_reason = false);
  bool constrained() const;
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
  bool Transition(std::string_view bytes, Trigger& next) const;
  bool Raw(std::int32_t token, std::string& out) const;
  bool Lexical(std::int32_t token, tokenizer::StreamDecoder& decoder, std::string& out) const;
  bool TerminalAllowed(std::int32_t token, bool can_stop) const;
  std::vector<tokenizer::SpecialSpan> Spans(std::size_t from, std::size_t lead,
                                            std::size_t consumed) const;
  std::expected<Selection, std::string> PlanOnce(std::int32_t selected) const;
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
