// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/calibration.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>
#include <system_error>
#include <utility>

#include "base/build_info.h"
#include "base/json.h"
#include "platform/files.h"
#include "platform/kept_files.h"
#include "providers/device_probe.h"

namespace jitllm::runtime {
namespace {

namespace fs = std::filesystem;
namespace json = jitllm::base::json;

std::string Errno(int error) { return std::strerror(error); }  // NOLINT(concurrency-mt-unsafe)

CalibrationRead Refused(std::string why) {
  return {.calibration = std::nullopt, .note = "refused: " + std::move(why), .refused = true};
}

bool IsId(std::string_view text) {
  return text.size() == 64 && std::ranges::all_of(text, [](char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
         });
}

// A number within a real key's range (config::ModelKeys).
bool InRealRange(const config::ModelKeySpec& spec, double v) {
  const bool low = spec.real_open ? v <= spec.real_least : v < spec.real_least;
  return std::isfinite(v) && !low && v <= spec.real_most;
}

// Keeps at most this many samples of a kind: the first ones.
constexpr std::size_t kMostSamples = 64;
// Whole prefills that measure the recompute cost a token: three of 8,192
// tokens or more, the slowest taken (prefill slows with depth; a short
// one's cost would rank long conversations too cheap to drop).
constexpr std::size_t kWholePrefills = 3;
constexpr std::uint32_t kLeastWholePrefill = 8192;
// Prefill speed counts wide chunks only: a short prompt's chunk is mostly
// fixed cost (124-token chats measured 150-200 tokens a second against
// about 900 for full chunks, DeepSeek on a GB10).
constexpr std::uint32_t kLeastChunkRows = 1024;

void Keep(std::vector<double>& samples, double value) {
  if (std::isfinite(value) && value > 0 && samples.size() < kMostSamples) {
    samples.push_back(value);
  }
}

}  // namespace

std::string MeasuredWith(const ModelSettings& s) {
  const auto form = [&] {
    switch (s.wave_form.value) {
      case config::WaveForm::kSpeculative:
        return "speculative";
      case config::WaveForm::kPlain:
        return "plain";
      case config::WaveForm::kAuto:
        break;
    }
    return "auto";
  };
  std::string measured = std::format(
      "speculation={} draft_rows={} prefill_chunk={} max_slots={} prefill_outa_hca={}/{} "
      "wave_form={} prefill_chunk_override={} max_slots_override={} context={}",
      s.speculation.value, s.draft_rows.value, s.prefill_chunk.value, s.max_slots.value,
      s.prefill_outa_hca.value, s.prefill_outa_hca_partial.value, form(),
      s.prefill_chunk.source == SettingSource::kOverride,
      s.max_slots.source == SettingSource::kOverride, s.context.value);
  if (s.architecture == "deepseek4") {
    measured += " wave_costs=[";
    for (std::size_t i = 0; i < s.wave_costs.value.size(); ++i) {
      measured += std::format("{}{}", i == 0 ? "" : ",", s.wave_costs.value[i]);
    }
    measured +=
        std::format("] wave_costs_override={} wave_costs_override_count={}",
                    s.wave_costs.source == SettingSource::kOverride, s.wave_costs_override_count);
  }
  if (s.architecture == "qwen4exp") {
    measured += std::format(
        " wave_lanes={} draft_vocab={} shared_wave_depth={} draft_wave_max={} "
        "wave_read_align={} depth_cost_ratio={} depth_cost_ratio_override={}",
        s.wave_lanes.value, s.draft_vocab.value, s.shared_wave_depth.value, s.draft_wave_max.value,
        s.wave_read_align.value, s.depth_cost_ratio.value,
        s.depth_cost_ratio.source == SettingSource::kOverride);
  }
  if (s.gemma31_production) {
    measured += " gemma31_production=true";
  }
  if (s.gemma26_production) {
    measured += " gemma26_production=true";
  }
  return measured;
}

std::string BuildIdentity() {
  const base::BuildInfo& build = base::GetBuildInfo();
  std::string identity(build.version);
  if (build.commit.empty() || build.modified) {
    if (const auto stamp = platform::RunningExecutableStamp()) {
      identity +=
          std::format(" (executable {} bytes, modified {} ns)", stamp->size, stamp->modified_ns);
    }
  }
  return identity;
}

CalibrationKey CalibrationKeyOf(const config::ModelEntry& entry,
                                const ModelSettings& uncalibrated) {
  static const std::string device = providers::DeviceIdentity("/");
  static const std::string build = BuildIdentity();
  return {.artifact = entry.artifact.value_or(""),
          .drafter = entry.drafter.value_or(""),
          .device = device,
          .build = build,
          .settings = MeasuredWith(uncalibrated)};
}

fs::path CalibrationPath(const fs::path& state, std::string_view artifact) {
  return state / "calibration" / std::format("{}.json", artifact);
}

CalibrationRead ParseCalibration(std::string_view text, const CalibrationKey& key) {
  json::Limits limits;
  limits.max_bytes = kMaxCalibrationBytes;
  auto doc = json::Parse(text, limits);
  if (!doc) {
    return Refused(doc.error().ToString());
  }
  const json::Value root = doc->root();
  if (!root.is_object()) {
    return Refused("not a JSON object");
  }
  const auto text_of = [&](std::string_view name) -> std::optional<std::string> {
    const auto v = root.find(name);
    return v && v->is_string() ? std::optional(std::string(v->string())) : std::nullopt;
  };
  for (std::size_t i = 0; i < root.size(); ++i) {
    const std::string_view name = root.key(i);
    if (name != "format" && name != "artifact" && name != "drafter" && name != "device" &&
        name != "build" && name != "settings" && name != "values") {
      return Refused(std::format("an unknown member \"{}\"", name));
    }
  }
  if (text_of("format") != kCalibrationFormat) {
    return Refused(std::format("not a {} record", kCalibrationFormat));
  }
  const auto artifact = text_of("artifact");
  if (!artifact || !IsId(*artifact)) {
    return Refused("no artifact ID");
  }
  if (*artifact != key.artifact) {
    return Refused(std::format("it is artifact {}'s", *artifact));
  }
  const auto drafter_value = root.find("drafter");
  std::string drafter;
  if (drafter_value && drafter_value->is_string() && IsId(drafter_value->string())) {
    drafter = drafter_value->string();
  } else if (!drafter_value || !drafter_value->is_null()) {
    return Refused("its drafter is neither an artifact ID nor null");
  }
  const auto device = text_of("device");
  const auto build = text_of("build");
  const auto settings = text_of("settings");
  if (!device || !build || !settings) {
    return Refused("no device, build or settings");
  }
  const auto values = root.find("values");
  if (!values || !values->is_object()) {
    return Refused("no values");
  }
  Calibration c;
  for (std::size_t i = 0; i < values->size(); ++i) {
    const std::string_view name = values->key(i);
    const json::Value v = values->member(i);
    const config::ModelKeySpec* spec = config::FindModelKey(name);
    if (spec == nullptr || std::ranges::find(kCalibrationKeys, name) == kCalibrationKeys.end()) {
      return Refused(std::format("\"{}\" is not a calibrated setting", name));
    }
    const auto bad = [&] { return Refused(std::format("\"{}\" is out of its range", name)); };
    if (name == "wave_costs") {
      if (!v.is_array() || v.size() == 0 || v.size() > c.wave_costs.size()) {
        return bad();
      }
      for (std::size_t w = 0; w < v.size(); ++w) {
        const json::Value cost = v.at(w);
        if (cost.is_null()) {
          continue;
        }
        const auto real = cost.is_number() ? cost.float64() : std::nullopt;
        if (!real || !InRealRange(*spec, *real)) {
          return bad();
        }
        c.wave_costs[w] = real;
      }
      continue;
    }
    if (spec->type == config::ModelKeyType::kInteger) {
      const auto integer = v.is_integer() ? v.int64() : std::nullopt;
      if (!integer || *integer < spec->least || *integer > spec->most) {
        return bad();
      }
      const auto u = static_cast<std::uint32_t>(*integer);
      if (name == "prefill_chunk") {
        c.prefill_chunk = u;
      } else if (name == "max_slots") {
        c.max_slots = u;
      } else if (name == "prefill_floor_tok_s") {
        c.prefill_floor_tok_s = u;
      } else {
        c.decode_floor_tok_s = u;
      }
      continue;
    }
    const auto real = v.is_number() ? v.float64() : std::nullopt;
    if (!real || !InRealRange(*spec, *real)) {
      return bad();
    }
    (name == "recompute_ms_per_token" ? c.recompute_ms_per_token : c.depth_cost_ratio) = real;
  }
  // Measured for something else: not in force, and measured again.
  if (drafter != key.drafter || *device != key.device || *build != key.build ||
      *settings != key.settings) {
    return {.calibration = std::nullopt,
            .note = std::format("stale: measured with {}, build {}, {}{}", *device, *build,
                                *settings, drafter != key.drafter ? ", another drafter" : ""),
            .refused = false};
  }
  c.basis = std::format("measured on this machine ({}, build {})", key.device, key.build);
  return {.calibration = std::move(c), .note = "in force", .refused = false};
}

CalibrationRead ReadCalibration(const fs::path& state, const CalibrationKey& key, uid_t trusted) {
  const fs::path path = CalibrationPath(state, key.artifact);
  std::error_code error;
  if (!fs::exists(fs::symlink_status(path, error))) {
    return {.calibration = std::nullopt, .note = "none recorded", .refused = false};
  }
  auto text = ReadTrustedFile(path, trusted, kMaxCalibrationBytes);
  if (!text) {
    return Refused(text.error());
  }
  return ParseCalibration(*text, key);
}

std::string FormatCalibration(const Calibration& c, const CalibrationKey& key) {
  std::string out = std::format(R"({{"format":"{}","artifact":)", kCalibrationFormat);
  json::AppendQuoted(key.artifact, out);
  out += ",\"drafter\":";
  if (key.drafter.empty()) {
    out += "null";
  } else {
    json::AppendQuoted(key.drafter, out);
  }
  out += ",\"device\":";
  json::AppendQuoted(key.device, out);
  out += ",\"build\":";
  json::AppendQuoted(key.build, out);
  out += ",\"settings\":";
  json::AppendQuoted(key.settings, out);
  out += ",\"values\":{";
  bool first = true;
  const auto add = [&](std::string_view name, const std::string& value) {
    out += std::format("{}\"{}\":{}", first ? "" : ",", name, value);
    first = false;
  };
  const auto number = [](double v) { return std::format("{}", v); };
  if (c.prefill_chunk) {
    add("prefill_chunk", std::format("{}", *c.prefill_chunk));
  }
  if (c.max_slots) {
    add("max_slots", std::format("{}", *c.max_slots));
  }
  if (c.prefill_floor_tok_s) {
    add("prefill_floor_tok_s", std::format("{}", *c.prefill_floor_tok_s));
  }
  if (c.decode_floor_tok_s) {
    add("decode_floor_tok_s", std::format("{}", *c.decode_floor_tok_s));
  }
  if (c.recompute_ms_per_token) {
    add("recompute_ms_per_token", number(*c.recompute_ms_per_token));
  }
  // Up to the widest measured width; null where unmeasured.
  std::size_t widths = 0;
  for (std::size_t w = 0; w < c.wave_costs.size(); ++w) {
    if (c.wave_costs[w]) {
      widths = w + 1;
    }
  }
  if (widths != 0) {
    std::string costs = "[";
    for (std::size_t w = 0; w < widths; ++w) {
      costs += std::format("{}{}", w == 0 ? "" : ",",
                           c.wave_costs[w] ? number(*c.wave_costs[w]) : std::string("null"));
    }
    add("wave_costs", costs + "]");
  }
  if (c.depth_cost_ratio) {
    add("depth_cost_ratio", number(*c.depth_cost_ratio));
  }
  return out + "}}";
}

std::expected<void, std::string> WriteCalibration(const fs::path& state, const CalibrationKey& key,
                                                  const Calibration& calibration) {
  // A directory of this user's that only it can enter, the record replaced
  // whole and atomically (platform/kept_files.h, as D-105's records are).
  const fs::path dir = state / "calibration";
  const auto dfd = platform::OpenPrivateDirectory(-1, dir.c_str());
  if (!dfd) {
    return std::unexpected(std::format(
        "{}: {}", dir.string(),
        dfd.error() == EPERM ? std::string("not a directory of this user's that only it can enter")
                             : Errno(dfd.error())));
  }
  const std::string name = std::format("{}.json", key.artifact);
  const auto written =
      platform::ReplacePrivateFile(*dfd, name.c_str(), FormatCalibration(calibration, key) + "\n");
  (void)::close(*dfd);
  if (!written) {
    return std::unexpected(std::format("{}: {}", (dir / name).string(), Errno(written.error())));
  }
  return {};
}

void CalibrationSamples::PrefillChunk(std::uint32_t at, std::uint32_t rows, double seconds) {
  if (at < kShortContext && rows >= kLeastChunkRows && seconds > 0) {
    Keep(prefill_rates_, rows / seconds);
    fresh_ = true;
  }
}

void CalibrationSamples::Prefill(std::uint32_t end, double seconds) {
  if (end >= kLeastWholePrefill && seconds > 0) {
    Keep(recompute_, seconds * 1000.0 / end);
    fresh_ = true;
  }
}

void CalibrationSamples::DecodeStep(std::uint32_t at, double tokens, double seconds) {
  if (at < kShortContext && tokens > 0 && seconds > 0) {
    Keep(decode_rates_, tokens / seconds);
    fresh_ = true;
  }
}

void CalibrationSamples::Wave(std::uint32_t width, bool speculative, double seconds) {
  if (width >= 2 && width - 2 < verify_waves_.size()) {
    Keep((speculative ? verify_waves_ : plain_waves_)[width - 2], seconds);
    fresh_ = true;
  }
}

void CalibrationSamples::DraftStep(std::uint32_t depth, double seconds) {
  if (depth == 2 || depth == 3) {
    Keep(depth == 2 ? depth2_ : depth3_, seconds);
    fresh_ = true;
  }
}

std::optional<double> CalibrationSamples::Median(std::vector<double> samples) {
  if (samples.size() < kSamples) {
    return std::nullopt;
  }
  const auto middle = samples.begin() + static_cast<std::ptrdiff_t>(samples.size() / 2);
  std::ranges::nth_element(samples, middle);
  return *middle;
}

std::optional<Calibration> CalibrationSamples::Take(const Calibration& known) {
  if (!fresh_) {
    return std::nullopt;
  }
  fresh_ = false;
  Calibration c = known;
  bool changed = false;
  const auto floor_of = [](double rate, std::uint32_t most) {
    const double floor = std::floor(rate / kFloorMargin);
    return static_cast<std::uint32_t>(std::clamp(floor, 1.0, static_cast<double>(most)));
  };
  if (!c.prefill_floor_tok_s) {
    if (const auto rate = Median(prefill_rates_)) {
      c.prefill_floor_tok_s = floor_of(*rate, config::kMaxPrefillFloor);
      changed = true;
    }
  }
  if (!c.decode_floor_tok_s) {
    if (const auto rate = Median(decode_rates_)) {
      c.decode_floor_tok_s = floor_of(*rate, config::kMaxDecodeFloor);
      changed = true;
    }
  }
  if (!c.recompute_ms_per_token && recompute_.size() >= kWholePrefills) {
    // The slowest a token: prefill slows with depth, and the longest of
    // the prefills measured is the least unlike a long conversation's.
    const double cost = std::ranges::max(recompute_);
    if (cost > 0 && cost <= 1e6) {
      c.recompute_ms_per_token = cost;
      changed = true;
    }
  }
  for (std::size_t w = 0; w < c.wave_costs.size(); ++w) {
    if (c.wave_costs[w]) {
      continue;
    }
    const auto verify = Median(verify_waves_[w]);
    const auto plain = Median(plain_waves_[w]);
    if (verify && plain && *plain > 0 && *verify / *plain <= 1000) {
      c.wave_costs[w] = *verify / *plain;
      changed = true;
    }
  }
  if (!c.depth_cost_ratio) {
    const auto three = Median(depth3_);
    const auto two = Median(depth2_);
    if (three && two && *two > 0 && *three / *two <= 100) {
      c.depth_cost_ratio = *three / *two;
      changed = true;
    }
  }
  if (!changed) {
    return std::nullopt;
  }
  return c;
}

execution::AdaptiveWaveMode::Mode WaveCostExploration::Choose(
    execution::AdaptiveWaveMode::Mode selected, std::uint32_t width, bool sampled,
    const ModelSettings& settings, const Calibration& known) {
  if (sampled || settings.wave_form.value != config::WaveForm::kAuto || width < 2 ||
      width > execution::AdaptiveWaveMode::kMaxWidth) {
    return selected;
  }
  const std::size_t cost = width - 2;
  if (cost < settings.wave_costs_override_count || cost >= known.wave_costs.size() ||
      known.wave_costs[cost] || explored_[width] >= kWaves) {
    return selected;
  }
  const auto mode = explored_[width] % 2 == 0 ? execution::AdaptiveWaveMode::Mode::kSpeculative
                                              : execution::AdaptiveWaveMode::Mode::kPlain;
  ++explored_[width];
  return mode;
}

}  // namespace jitllm::runtime
