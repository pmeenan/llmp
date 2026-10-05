// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <iosfwd>

// Diagnostic-only interface linked by the manual helper. The reversible
// measurement overlay uses it; canonical engine source/objects do not.
namespace jitllm::benchmark::coarse {
enum class Phase : std::uint8_t {
  kWave,
  kPlacesCall,
  kPlacesBody,
  kInputs,
  kStateGrow,
  kStateDescribe,
  kStateDescribeBody,
  kStateAcquire,
  kStateRegister,
  kStateRegisterBody,
  kPlanHit,
  kPlanMiss,
  kGraphBuildBind,
  kPlanFirst,
  kPlacement,
  kPlanSecond,
  kPlanBind,
  kCoverage,
  kStage,
  kJobCompleted,
  kJobSubmit,
  kPublish,
  kHostRelease,
  kRefreshCall,
  kRefreshBody,
  kCohortHold,
  kProbeBusy,
  kProbeWait,
  kCount
};
struct Context {
  bool enabled = false;
  std::uint8_t ordinal = 0;
};
Context Current();
bool Begin();
bool Ordinal(std::uint32_t ordinal);
void End();
bool Report(std::ostream& output);
bool ClockProbe(std::ostream& output);
class Scope {
 public:
  Scope(Phase phase, Context context);
  explicit Scope(Phase phase) : Scope(phase, Current()) {}
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
  ~Scope();

 private:
  Phase phase_;
  Context context_;
  std::uint32_t owner_ = 0;
  std::int64_t wall_ = 0;
  std::int64_t cpu_ = 0;
};
}  // namespace jitllm::benchmark::coarse
