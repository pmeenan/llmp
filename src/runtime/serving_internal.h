// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_RUNTIME_SERVING_INTERNAL_H_
#define LLMP_RUNTIME_SERVING_INTERNAL_H_
#include "runtime/serving.h"
#include "runtime/serving_support.h"
namespace llmp::runtime::serving_detail {
// Diagnostic caps alone enable these clocks; ordinary serving adds no clock
// reads or scheduler calls. The driver owns every accumulator.
class SwapPhaseTimer {
 public:
  explicit SwapPhaseTimer(double* total)
      : total_(total), begin_(total != nullptr ? Clock::now() : Clock::time_point{}) {}
  SwapPhaseTimer(const SwapPhaseTimer&) = delete;
  SwapPhaseTimer& operator=(const SwapPhaseTimer&) = delete;
  ~SwapPhaseTimer() {
    if (total_ != nullptr) *total_ += serving_support::Seconds(Clock::now() - begin_);
  }

 private:
  double* total_;
  Clock::time_point begin_;
};

}  // namespace llmp::runtime::serving_detail
#endif  // LLMP_RUNTIME_SERVING_INTERNAL_H_
