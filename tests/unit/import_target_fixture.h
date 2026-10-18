// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_TESTS_UNIT_IMPORT_TARGET_FIXTURE_H_
#define LLMP_TESTS_UNIT_IMPORT_TARGET_FIXTURE_H_
#include "base/check.h"
#include "dsv4_fixture.h"
#include "families/gemma.h"
#include "import/normalize.h"
#include "spec/mappings/dsv4.h"
#include "spec/mappings/gemma4.h"
#include "spec_fixtures.h"

namespace llmp::test_support {
inline import::NormalizationTarget GemmaImportTarget(std::uint32_t size) {
  import::NormalizationTarget target;
  target.mapping = &spec::mappings::Gemma4Mapping();
  target.source = Gemma4Source(size);
  auto values =
      families::Gemma4Values(size == 26 ? families::Gemma4_26BA4B() : families::Gemma4_31B());
  base::Check(values.has_value(), "target fixture values");
  auto model = spec::Synthesize(*target.mapping, target.source, *values);
  base::Check(model.has_value(), "target fixture spec");
  target.model = std::move(*model);
  return target;
}
inline import::NormalizationTarget Dsv4ImportTarget() {
  import::NormalizationTarget target;
  target.mapping = &spec::mappings::Dsv4Mapping();
  const auto& p = families::Dsv4Flash();
  target.source = dsv4::SourceOf("deepseek4", dsv4::Resources(p), *target.mapping);
  auto values = families::Dsv4Values(p);
  base::Check(values.has_value(), "target fixture values");
  auto model = spec::Synthesize(*target.mapping, target.source, *values);
  base::Check(model.has_value(), "target fixture spec");
  target.model = std::move(*model);
  return target;
}
inline std::expected<import::NormalizedPlan, import::Error> NormalizeWithFixtureTarget(
    import::Checkpoint source, import::NormalizationLimits limits) {
  if (source.architecture == "gemma4-assistant") {
    const auto* width = source.FindValue("gemma4-assistant.embedding_length_out");
    const auto* value = width ? std::get_if<std::int64_t>(width) : nullptr;
    auto target = GemmaImportTarget(value && *value == 5376 ? 31 : 26);
    return import::Normalize(std::move(source), limits, &target);
  }
  if (source.architecture == "dflash") {
    auto target = Dsv4ImportTarget();
    return import::Normalize(std::move(source), limits, &target);
  }
  return import::Normalize(std::move(source), limits);
}
}  // namespace llmp::test_support
#endif  // LLMP_TESTS_UNIT_IMPORT_TARGET_FIXTURE_H_
