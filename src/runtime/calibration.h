// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Calibration on this machine (D-103's second layer;
// docs/runtime-serving.md#calibration): a model's measured speed trades,
// recorded under the state role and used by registration whenever the
// record was measured for the same artifacts, device, driver and build.
//
// The record: `<state>/calibration/<artifact ID>.json`, one JSON object,
//
//   {"format":"jitllm-model-calibration-v1",
//    "artifact":"<64 hex>", "drafter":"<64 hex>" or null,
//    "device":"NVIDIA GB10 sm_121, driver 580.95.05, CUDA 13.0",
//    "build":"0.3.0-dev.12+g1234567",
//    "values":{"prefill_floor_tok_s":333, "wave_costs":[2.1, ...], ...}}
//
// whose values are settings' keys (config::ModelKeys) of the measured kind
// (CalibrationKeys), each within its key's range. A record measured for
// another device, driver, build or drafter is stale: not used, and
// replaced when the model is measured again. One that is not this format,
// names another artifact, or holds an unknown key or a value out of range
// is refused (logged, not used), and replaced the same way. Records are
// written atomically (a new file renamed over the old, both synced), 0600
// in a 0700 directory, and read under the state role's trust rules.
//
// What is measured, passively, from a model's first uses
// (CalibrationSamples): its prefill and decode speeds at short context
// (the floors are a third of them, the margin for depth), prefill's cost a
// token (the reclaim order's recompute cost),
// DeepSeek's draft-verify against plain wave times at each width, and
// Qwen3.8's three-draft against two-draft step times. A value is recorded
// once enough samples agree; it is in force from the next registration,
// never mid-service, so a running schedule never changes with wall time.
// Prefill chunk rows and the request-slot knee need controlled runs and
// keep their fallbacks; kernel schedules are later work.

#ifndef JITLLM_RUNTIME_CALIBRATION_H_
#define JITLLM_RUNTIME_CALIBRATION_H_

#include <sys/types.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "config/node_config.h"
#include "runtime/model_settings.h"

namespace jitllm::runtime {

inline constexpr std::string_view kCalibrationFormat = "jitllm-model-calibration-v1";
inline constexpr std::size_t kMaxCalibrationBytes = std::size_t{64} << 10U;
// The settings a calibration may hold.
inline constexpr std::array<std::string_view, 7> kCalibrationKeys = {
    "prefill_chunk",          "max_slots",  "prefill_floor_tok_s", "decode_floor_tok_s",
    "recompute_ms_per_token", "wave_costs", "depth_cost_ratio"};

// What a record is for.
struct CalibrationKey {
  std::string artifact;
  std::string drafter;   // empty: none
  std::string device;    // providers::DeviceIdentity
  std::string build;     // base::BuildInfo's version (BuildIdentity)
  std::string settings;  // MeasuredWith: the settings its measurements depend on
};

// The settings a measurement depends on, as resolved without calibration
// (override, derived or fallback): speculation, draft rows, prefill chunk
// rows, request slots, the output-A/HCA prefill and the wave form; Qwen's
// wave lanes, effective draft vocabulary, shared draft cap, joined-draft
// limit, wave read alignment and depth cost ratio. Another configuration
// of them makes a record stale. The depth ratio here is resolved without
// calibration, so the recorded ratio does not invalidate its own key.
// Override presence for the chunk, slot cap and depth ratio also matters:
// setting the fallback value explicitly bypasses a calibrated value.
std::string MeasuredWith(const ModelSettings& s);

// This build's identity: its version, and for a tree without Git metadata
// or with uncommitted changes (whose version does not change between
// builds: "0.1.0-dev+unknown") also the executable's size and modification
// time, so a rebuilt binary is measured again.
std::string BuildIdentity();

// The key of a configured model's record on this host: its artifact and
// drafter, this host's device and driver (providers::DeviceIdentity, probed
// once a process), this build (BuildIdentity) and its settings resolved
// without calibration (MeasuredWith).
CalibrationKey CalibrationKeyOf(const config::ModelEntry& entry, const ModelSettings& uncalibrated);

// The record's path for `artifact` under the state role.
std::filesystem::path CalibrationPath(const std::filesystem::path& state,
                                      std::string_view artifact);

// A record as found: the calibration in force, if any, and otherwise why
// none ("none recorded", "stale: ...", or "refused: ..." for a corrupt or
// foreign one, which `refused` marks).
struct CalibrationRead {
  std::optional<Calibration> calibration;
  std::string note;
  bool refused = false;
};

// A record's text, checked against `key` (stale) and its format (refused).
CalibrationRead ParseCalibration(std::string_view text, const CalibrationKey& key);
// The record of `key.artifact` under `state`, read under the trust rules
// (root's or `trusted`'s, writable by nobody else).
CalibrationRead ReadCalibration(const std::filesystem::path& state, const CalibrationKey& key,
                                uid_t trusted);
// The record's text for `calibration`, measured for `key`.
std::string FormatCalibration(const Calibration& calibration, const CalibrationKey& key);
// Writes it atomically, creating `<state>/calibration` (0700) if absent.
std::expected<void, std::string> WriteCalibration(const std::filesystem::path& state,
                                                  const CalibrationKey& key,
                                                  const Calibration& calibration);

// A model's first uses, measured: each kind of sample kept until its value
// is known. Not thread-safe: the driver's thread alone records.
class CalibrationSamples {
 public:
  // A value is measured from at least this many samples (their median).
  static constexpr std::size_t kSamples = 8;
  // Speeds count at short context only: chunks and steps from before it.
  static constexpr std::uint32_t kShortContext = 32768;
  // The floors are the measured speeds over this margin (for depth).
  static constexpr double kFloorMargin = 3.0;

  // A prefill chunk of `rows` from position `at` in `seconds`.
  void PrefillChunk(std::uint32_t at, std::uint32_t rows, double seconds);
  // A whole prefill from position 0 to `end` in `seconds`: what dropping
  // its conversation would cost again, a token.
  void Prefill(std::uint32_t end, double seconds);
  // A decode step or wave from position `at` (its members' least) that
  // committed `tokens` a request, on average, in `seconds`: each request's
  // decode speed, peers sharing the wave included.
  void DecodeStep(std::uint32_t at, double tokens, double seconds);
  // A wave of `width` requests in either form, in `seconds`.
  void Wave(std::uint32_t width, bool speculative, double seconds);
  // A lone speculative step of `depth` drafts in `seconds`.
  void DraftStep(std::uint32_t depth, double seconds);

  // The values measured so far over `known` (the record's, or none):
  // nullopt when nothing is new since the last Take.
  std::optional<Calibration> Take(const Calibration& known);
  // Every sample not yet taken dropped: a hang's cancelled unit (D-102) may
  // have been timed among them, and a hang's wait is no measure of the
  // machine. Measuring starts over.
  void Forget() { *this = CalibrationSamples{}; }

 private:
  static std::optional<double> Median(std::vector<double> samples);
  std::vector<double> prefill_rates_;
  std::vector<double> decode_rates_;
  std::vector<double> recompute_;
  std::array<std::vector<double>, config::kMaxWaveCostWidths> verify_waves_{};
  std::array<std::vector<double>, config::kMaxWaveCostWidths> plain_waves_{};
  std::vector<double> depth2_;
  std::vector<double> depth3_;
  bool fresh_ = false;  // a sample since the last Take
};

}  // namespace jitllm::runtime

#endif  // JITLLM_RUNTIME_CALIBRATION_H_
