// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_RUNTIME_LLM_INTERNAL_H_
#define LLMP_RUNTIME_LLM_INTERNAL_H_
#include <utility>
namespace llmp::runtime::llm_detail {
// Marks history growth while prompt/checkpoint/session work owns the branch.
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

}  // namespace llmp::runtime::llm_detail
#endif  // LLMP_RUNTIME_LLM_INTERNAL_H_
