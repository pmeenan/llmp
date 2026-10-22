// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "import/sampling_defaults.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>
#include <variant>

#include "base/json.h"
#include "base/sha256.h"
#include "import/files.h"

namespace llmp::import {
namespace {
namespace json = base::json;
constexpr std::string_view kGeneration = "generation_config.json";
constexpr std::uint64_t kMaxDocument = json::Limits{}.max_bytes;

std::unexpected<Error> Bad(std::string_view reason, Rule rule = Rule::kConfig) {
  return std::unexpected(Error{.rule = rule, .reason = reason});
}

std::optional<double> Real(const MetaValue* value) {
  if (!value) return std::nullopt;
  if (const auto* real = std::get_if<double>(value)) return *real;
  if (const auto* integer = std::get_if<std::int64_t>(value)) return static_cast<double>(*integer);
  return std::nullopt;
}
std::optional<std::int64_t> Integer(const MetaValue* value) {
  if (value)
    if (const auto* integer = std::get_if<std::int64_t>(value)) return *integer;
  return std::nullopt;
}

void Merge(spec::SamplingDefaults& out, std::optional<double> temperature,
           std::optional<double> top_p, std::optional<std::int64_t> top_k,
           std::optional<double> min_p) {
  if (temperature && std::isfinite(*temperature) && *temperature >= 0 && *temperature <= 2)
    out.temperature = *temperature;
  if (top_p && std::isfinite(*top_p) && static_cast<float>(*top_p) > 0 && *top_p <= 1)
    out.top_p = *top_p;
  if (top_k && *top_k >= 0 && *top_k <= std::numeric_limits<std::int32_t>::max())
    out.top_k = *top_k;
  if (min_p && std::isfinite(*min_p) && *min_p >= 0 && *min_p <= 1) out.min_p = *min_p;
}
}  // namespace

std::expected<SamplingDefaultsCost, Error> SamplingDefaultsBound(const Checkpoint& source) {
  SamplingDefaultsCost cost;
  if (source.container == "gguf") return cost;
  bool found = false;
  for (const auto& file : source.files) {
    if (file.name != kGeneration) continue;
    if (found) return Bad("duplicate generation_config.json source");
    found = true;
    if (file.bytes > kMaxDocument)
      return Bad("generation_config.json exceeds the JSON document bound", Rule::kBounds);
    // JSON nodes, child indices, decoded strings, sorting temporaries and
    // their vector growth fit within 128 bytes per input byte; the fixed
    // term covers the depth-bounded parser stack and allocation overhead.
    // kMaxDocument makes both products nonoverflowing on supported targets.
    cost.metadata_bytes = 65536 + file.bytes * 128;
    cost.work_units = 4096 + file.bytes * 128;
  }
  return cost;
}

std::expected<spec::SamplingDefaults, Error> ReadSamplingDefaults(Checkpoint& source,
                                                                  SamplingDefaultsCost prepaid,
                                                                  std::string_view nested_prefix) {
  const auto cost = SamplingDefaultsBound(source);
  if (!cost) return std::unexpected(cost.error());
  if (prepaid.metadata_bytes < cost->metadata_bytes || prepaid.work_units < cost->work_units)
    return Bad("sampling defaults parsing is not prepaid", Rule::kBounds);
  spec::SamplingDefaults out;
  if (source.container == "gguf") {
    Merge(out, Real(source.FindValue("general.sampling.temp")),
          Real(source.FindValue("general.sampling.top_p")),
          Integer(source.FindValue("general.sampling.top_k")),
          Real(source.FindValue("general.sampling.min_p")));
    return out;
  }
  Merge(out, Real(source.FindValue("temperature")), Real(source.FindValue("top_p")),
        Integer(source.FindValue("top_k")), Real(source.FindValue("min_p")));
  std::optional<bool> sample;
  if (const auto* value = source.FindValue("do_sample"))
    if (const auto* flag = std::get_if<bool>(value)) sample = *flag;
  // Decoder defaults fill unset outer defaults. The mapping supplies its
  // container nesting prefix; no architecture is named in this reader.
  if (!nested_prefix.empty()) {
    if (nested_prefix.size() > spec::kMaxText || !nested_prefix.ends_with('.'))
      return Bad("sampling defaults have an invalid nesting prefix", Rule::kBounds);
    const auto nested_value = [&](std::string_view key) {
      std::array<char, spec::kMaxText + 32> text{};
      auto end = std::copy(nested_prefix.begin(), nested_prefix.end(), text.begin());
      end = std::copy(key.begin(), key.end(), end);
      return source.FindValue({text.data(), static_cast<std::size_t>(end - text.begin())});
    };
    spec::SamplingDefaults nested;
    Merge(nested, Real(nested_value("temperature")), Real(nested_value("top_p")),
          Integer(nested_value("top_k")), Real(nested_value("min_p")));
    if (!out.temperature) out.temperature = nested.temperature;
    if (!out.top_p) out.top_p = nested.top_p;
    if (!out.top_k) out.top_k = nested.top_k;
    if (!out.min_p) out.min_p = nested.min_p;
    if (!sample)
      if (const auto* value = nested_value("do_sample"))
        if (const auto* flag = std::get_if<bool>(value)) sample = *flag;
  }
  for (std::size_t i = 0; i < source.files.size(); ++i) {
    auto& file = source.files[i];
    if (file.name != kGeneration) continue;
    auto opened = OpenCheckpointFile(file.path, i);
    if (!opened) return std::unexpected(opened.error());
    if (opened->device != file.device || opened->inode != file.inode || opened->bytes != file.bytes)
      return Bad("generation_config.json changed identity or size after reading", Rule::kIo);
    auto text = ReadDocument(*opened, kMaxDocument, i);
    if (!text) return std::unexpected(text.error());
    if (file.parsed.bytes > file.bytes ||
        (file.parsed.bytes != 0 &&
         base::Sha256().Update(std::string_view(*text).substr(0, file.parsed.bytes)).Finish() !=
             file.parsed.sha256))
      return Bad("generation_config.json changed after reading", Rule::kIo);
    auto doc = json::Parse(*text);
    if (!doc || !doc->root().is_object()) return Bad("generation_config.json is not a JSON object");
    const auto root = doc->root();
    const auto real = [&](std::string_view key) -> std::optional<double> {
      const auto value = root.find(key);
      return value && value->is_number() ? value->float64() : std::nullopt;
    };
    std::optional<std::int64_t> top_k;
    if (const auto value = root.find("top_k"); value && value->is_integer()) top_k = value->int64();
    Merge(out, real("temperature"), real("top_p"), top_k, real("min_p"));
    if (const auto value = root.find("do_sample"); value && value->is_bool())
      sample = value->boolean();
    file.parsed = {file.bytes, base::Sha256().Update(*text).Finish()};
  }
  if (sample && !*sample) out.temperature = 0.0;
  return out;
}
}  // namespace llmp::import
