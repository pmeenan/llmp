// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_CHAT_TOOL_SAMPLING_H_
#define LLMP_CHAT_TOOL_SAMPLING_H_
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "chat/tool_calls.h"
#include "tokenizer/tokenizer.h"
namespace llmp::chat {
// Request-local accepted-token cursor. A speculative judge copies it once per
// unit, then advances only accepted candidates. Vocabulary probes borrow the
// pending argument bytes; they never clone argument payloads.
class ToolSampling {
 public:
  ToolSampling(const tokenizer::Tokenizer& tokenizer, ToolCallFormat format, ToolRequest request,
               base::json::Limits limits, bool reasoning, std::span<const std::int32_t> terminals,
               std::span<const std::int32_t> route_stops = {});
  bool constrained() const;
  // A request-local sampling refusal, distinct from invalid native logits.
  void RefuseCandidates() { refused_candidates_ = true; }
  bool refused_candidates() const { return refused_candidates_; }
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

 private:
  struct Context {
    const tokenizer::Tokenizer* tokenizer;
    ToolCallFormat format;
    ToolRequest request;
    std::string prefix;
    std::vector<std::int32_t> terminals;
  };
  struct Trigger {
    bool waiting = false;
    std::size_t forced = 0;
    std::string tail;
  };
  bool Transition(std::string_view bytes, Trigger& next) const;
  bool Decode(std::int32_t token, std::string& out) const;
  bool TerminalAllowed(std::int32_t token, bool can_stop) const;
  std::shared_ptr<const Context> context_;
  ToolInterpreter lexical_;
  Trigger trigger_;
  bool last_reasoning_boundary_ = false;
  bool refused_candidates_ = false;
  std::size_t tokens_left_ = 0, context_left_ = 0;
  std::vector<std::int32_t> certain_;
};
}  // namespace llmp::chat
#endif
