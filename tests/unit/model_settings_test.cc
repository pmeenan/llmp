// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A model's settings in three layers (D-103, runtime/model_settings.h):
// each from the artifact's facts, a calibration or an override, else its
// fallback, with its source; refused overrides; the reasoning markers; and
// the listing against the configuration's table-driven schema.

#include "runtime/model_settings.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <vector>

#include "artifact/artifact.h"
#include "artifact_writer.h"
#include "config/node_config.h"
#include "runtime/approvals.h"
#include "runtime/component_admission.h"
#include "spec/mapping.h"
#include "spec/spec.h"

namespace {

using ::llmp::config::ModelEntry;
using ::llmp::runtime::ArtifactFacts;
using ::llmp::runtime::Calibration;
using ::llmp::runtime::ModelSettings;
using ::llmp::runtime::ResolveReasoning;
using ::llmp::runtime::ResolveSettings;
using ::llmp::runtime::SettingSource;
using ::testing::Contains;
using ::testing::HasSubstr;

ModelEntry Model(std::string name, bool drafter = false) {
  ModelEntry entry;
  entry.name = std::move(name);
  entry.artifact = std::string(64, 'a');
  if (drafter) {
    entry.drafter = std::string(64, 'b');
  }
  return entry;
}

ArtifactFacts Facts(std::string architecture, std::string drafter = {}) {
  ArtifactFacts facts;
  facts.architecture = std::move(architecture);
  facts.drafter_mtp = drafter == "qwen4exp-mtp" || drafter == "qwen4exp_mtp";
  facts.drafter_architecture = std::move(drafter);
  if (facts.architecture == "deepseek4") {
    facts.context_ceiling = 1048576;
  } else if (facts.architecture == "qwen4exp") {
    facts.context_ceiling = 262144;
    facts.vocabulary = 248320;
  } else if (facts.architecture == "gemma4") {
    facts.context_ceiling = 262144;
  } else if (facts.architecture == "gemma3") {
    facts.context_ceiling = 8448;
  } else if (facts.architecture == "gemma2") {
    facts.context_ceiling = 8192;
  }
  return facts;
}

ModelSettings Resolved(const ModelEntry& entry, const ArtifactFacts& facts,
                       const Calibration* calibration = nullptr, bool plain = false) {
  auto s = ResolveSettings(entry, facts, calibration, plain);
  EXPECT_TRUE(s.has_value()) << (s ? "" : s.error());
  return s ? *s : ModelSettings{};
}

// With nothing kept and nothing set, every setting is its fallback (the
// constants M3 measured), except what the configuration itself decides.
TEST(ModelSettings, BoundSpecSelectionDoesNotDependOnCalibration) {
  auto entry = Model("shared", true);
  auto facts = Facts("gemma4", "gemma4-assistant");
  facts.target_spec = llmp::base::Sha256Digest{};
  facts.composed_spec = llmp::base::Sha256Digest{};
  (*facts.target_spec)[0] = 1;
  (*facts.composed_spec)[0] = 2;
  const auto uncalibrated = ResolveSettings(entry, facts, nullptr, false);
  ASSERT_TRUE(uncalibrated) << (uncalibrated ? "" : uncalibrated.error());
  EXPECT_EQ(uncalibrated->bound_spec, facts.composed_spec);
  Calibration calibration;
  calibration.prefill_chunk = 64;
  const auto calibrated = ResolveSettings(entry, facts, &calibration, false);
  ASSERT_TRUE(calibrated) << (calibrated ? "" : calibrated.error());
  EXPECT_EQ(calibrated->bound_spec, uncalibrated->bound_spec);
  const auto plain = ResolveSettings(entry, facts, &calibration, true);
  ASSERT_TRUE(plain) << (plain ? "" : plain.error());
  EXPECT_EQ(plain->bound_spec, facts.target_spec);
  entry.overrides["speculation"] = false;
  const auto disabled = ResolveSettings(entry, facts, &calibration, false);
  ASSERT_TRUE(disabled) << (disabled ? "" : disabled.error());
  EXPECT_EQ(disabled->bound_spec, facts.target_spec);
  entry.overrides.erase("speculation");
  facts.composed_spec.reset();
  facts.composed_spec_error = "unqualified companion";
  EXPECT_FALSE(ResolveSettings(entry, facts, nullptr, false));
  EXPECT_TRUE(ResolveSettings(entry, facts, nullptr, true));
  facts.target_spec.reset();
  const auto legacy = ResolveSettings(entry, facts, nullptr, false);
  ASSERT_TRUE(legacy) << (legacy ? "" : legacy.error());
  EXPECT_FALSE(legacy->bound_spec);
}

TEST(ModelSettings, LossyCacheChoiceRequiresMatchingFactsAndIsNotCalibrated) {
  auto entry = Model("shared");
  auto facts = Facts("gemma2");
  facts.target_spec = llmp::base::Sha256Digest{};
  const auto baseline = ResolveSettings(entry, facts, nullptr, false);
  ASSERT_TRUE(baseline);
  EXPECT_EQ(baseline->kv_cache_encoding.value, "f16");
  entry.overrides["kv_cache_encoding"] = std::string("q8_0");
  EXPECT_FALSE(ResolveSettings(entry, facts, nullptr, false));
  facts.kv_cache_encoding = "q8_0";
  (*facts.target_spec)[0] = 1;
  Calibration calibration;
  calibration.prefill_chunk = 64;
  const auto configured = ResolveSettings(entry, facts, &calibration, false);
  ASSERT_TRUE(configured) << configured.error();
  EXPECT_EQ(configured->kv_cache_encoding.value, "q8_0");
  EXPECT_EQ(configured->kv_cache_encoding.source, SettingSource::kOverride);
  EXPECT_EQ(configured->bound_spec, facts.target_spec);
  EXPECT_NE(configured->bound_spec, baseline->bound_spec);
  entry.drafter = std::string(64, 'b');
  EXPECT_FALSE(ResolveSettings(entry, facts, nullptr, true));
  entry.drafter.reset();
  entry.overrides.erase("kv_cache_encoding");
  EXPECT_FALSE(ResolveSettings(entry, facts, &calibration, false));
  auto image_entry = Model("image");
  image_entry.artifact.reset();
  image_entry.composition = std::string(64, 'c');
  ArtifactFacts image_facts;
  image_facts.composition = true;
  const auto image = ResolveSettings(image_entry, image_facts, nullptr, false);
  ASSERT_TRUE(image);
  for (const auto& line : image->Lines()) EXPECT_NE(line.key, "kv_cache_encoding");
  image_entry.overrides["kv_cache_encoding"] = std::string("f16");
  EXPECT_FALSE(ResolveSettings(image_entry, image_facts, nullptr, false));
}

TEST(ModelSettings, DrafterCompositionAdmitsQ80WithComposedSpec) {
  auto entry = Model("ds", true);
  entry.overrides["kv_cache_encoding"] = std::string("q8_0");
  auto facts = Facts("deepseek4", "dflash");
  facts.kv_cache_encoding = "q8_0";
  facts.target_spec = llmp::base::Sha256Digest{};
  (*facts.target_spec)[0] = 1;
  // Without composed_spec, drafter with Q8_0 fails.
  EXPECT_FALSE(ResolveSettings(entry, facts, nullptr, false));
  // With composed_spec, it succeeds.
  facts.composed_spec = llmp::base::Sha256Digest{};
  (*facts.composed_spec)[0] = 2;
  const auto composed = ResolveSettings(entry, facts, nullptr, false);
  ASSERT_TRUE(composed) << composed.error();
  EXPECT_EQ(composed->kv_cache_encoding.value, "q8_0");
  EXPECT_EQ(composed->bound_spec, facts.composed_spec);
}

TEST(ModelSettings, Gemma4AssistantDrafterCompositionResolvesSettingsAndPlainMode) {
  auto entry = Model("gemma4_asst", true);
  auto facts = Facts("gemma4", "gemma4-assistant");
  facts.target_spec = llmp::base::Sha256Digest{};
  (*facts.target_spec)[0] = 42;
  facts.composed_spec = llmp::base::Sha256Digest{};
  (*facts.composed_spec)[0] = 84;

  const auto composed = ResolveSettings(entry, facts, nullptr, false);
  ASSERT_TRUE(composed) << (composed ? "" : composed.error());
  EXPECT_TRUE(composed->speculation.value);
  EXPECT_EQ(composed->bound_spec, facts.composed_spec);
  EXPECT_EQ(composed->architecture, "gemma4");
  EXPECT_EQ(composed->drafter, true);

  // Plain mode disables speculation and uses target_spec.
  const auto plain = ResolveSettings(entry, facts, nullptr, true);
  ASSERT_TRUE(plain) << (plain ? "" : plain.error());
  EXPECT_FALSE(plain->speculation.value);
  EXPECT_EQ(plain->bound_spec, facts.target_spec);

  // Override speculation=false also selects target_spec.
  entry.overrides["speculation"] = false;
  const auto no_spec = ResolveSettings(entry, facts, nullptr, false);
  ASSERT_TRUE(no_spec) << (no_spec ? "" : no_spec.error());
  EXPECT_FALSE(no_spec->speculation.value);
  EXPECT_EQ(no_spec->bound_spec, facts.target_spec);

  // When composed_spec is absent and error is set, resolution fails unless plain is requested.
  entry.overrides.erase("speculation");
  facts.composed_spec.reset();
  facts.composed_spec_error = "unqualified drafter";
  EXPECT_FALSE(ResolveSettings(entry, facts, nullptr, false));
  EXPECT_TRUE(ResolveSettings(entry, facts, nullptr, true));
}

TEST(ModelSettings, DflashCompositionResolvesSettingsAndCapabilities) {
  auto entry = Model("ds_dflash", true);
  auto facts = Facts("deepseek4", "dflash");
  facts.target_spec = llmp::base::Sha256Digest{};
  (*facts.target_spec)[0] = 11;
  facts.composed_spec = llmp::base::Sha256Digest{};
  (*facts.composed_spec)[0] = 22;

  const auto composed = ResolveSettings(entry, facts, nullptr, false);
  ASSERT_TRUE(composed) << (composed ? "" : composed.error());
  EXPECT_TRUE(composed->speculation.value);
  EXPECT_EQ(composed->bound_spec, facts.composed_spec);
  EXPECT_EQ(composed->draft_rows.value, 3U);
  EXPECT_EQ(composed->wave_form.value, llmp::config::WaveForm::kAuto);

  // Plain mode selects target_spec and disables speculation.
  const auto plain = ResolveSettings(entry, facts, nullptr, true);
  ASSERT_TRUE(plain) << (plain ? "" : plain.error());
  EXPECT_FALSE(plain->speculation.value);
  EXPECT_EQ(plain->bound_spec, facts.target_spec);
}

TEST(ModelSettings, FallbacksWhenTheArtifactSaysNothing) {
  const ModelSettings s = Resolved(Model("ds"), Facts("deepseek4"));
  EXPECT_EQ(s.context.value, 262144U);
  EXPECT_EQ(s.context.source, SettingSource::kFallback);
  EXPECT_FALSE(s.speculation.value);
  EXPECT_EQ(s.speculation.source, SettingSource::kDerived);  // no drafter
  EXPECT_EQ(s.prefill_chunk.value, llmp::runtime::RecipeOf("deepseek4")->prefill_rows);
  EXPECT_EQ(s.prefill_chunk.source, SettingSource::kFallback);
  EXPECT_EQ(s.max_slots.value, 4U);
  EXPECT_EQ(s.prefill_floor_tok_s.value, 100U);
  EXPECT_EQ(s.decode_floor_tok_s.value, 5U);
  EXPECT_EQ(s.recompute_ms_per_token.value, 1.4);
  EXPECT_EQ(s.temperature.value, 1.0);
  EXPECT_EQ(s.temperature.source, SettingSource::kFallback);
  EXPECT_EQ(s.top_k.value, 0U);
  EXPECT_EQ(s.draft_rows.value, 3U);
  EXPECT_EQ(s.wave_form.value, llmp::config::WaveForm::kAuto);
  // DeepSeek's recipe's measured costs (spec/mappings/dsv4.cc), and the
  // state its joined waves read rounded as Qwen3.8's.
  EXPECT_EQ(s.wave_costs.value, (std::vector<double>{2.08, 2.52, 2.81, 2.40, 2.12, 2.21, 2.23}));
  EXPECT_EQ(s.wave_read_align.value, 2048U);
  EXPECT_TRUE(s.wave_lanes.value);
  EXPECT_EQ(s.wave_costs_override_count, 0U);
  EXPECT_TRUE(s.ignored.empty());

  const ModelSettings q = Resolved(Model("q"), Facts("qwen4exp"));
  EXPECT_EQ(q.context.value, 262144U);
  EXPECT_EQ(q.prefill_chunk.value, llmp::runtime::RecipeOf("qwen4exp")->prefill_rows);
  EXPECT_EQ(q.draft_vocab.value, 65536U);
  EXPECT_EQ(q.depth_cost_ratio.value, 1.16);
  EXPECT_EQ(q.shared_wave_depth.value, 2U);
  EXPECT_EQ(q.draft_wave_max.value, 2U);
  EXPECT_EQ(q.wave_read_align.value, 2048U);
  EXPECT_FALSE(q.wave_lanes.value);
}

TEST(ModelSettings, GemmaCohortPreferencesDoNotManufactureOrRestrictPhysicalCapabilities) {
  for (const auto* architecture : {"gemma2", "gemma3", "gemma4"}) {
    SCOPED_TRACE(architecture);
    auto facts = Facts(architecture);
    facts.trained_context = 262144;
    facts.trained_context_from = "stored context";
    auto entry = Model("gemma");
    auto defaults = ResolveSettings(entry, facts, nullptr, false);
    ASSERT_TRUE(defaults);
    EXPECT_EQ(defaults->max_slots.value, 4U);
    EXPECT_FALSE(defaults->speculation.value);
    for (const auto owners : {1, 3, 4, 8, 16}) {
      entry.overrides["max_slots"] = std::int64_t{owners};
      entry.overrides["prefill_chunk"] = std::int64_t{256};
      auto settings = ResolveSettings(entry, facts, nullptr, false);
      ASSERT_TRUE(settings);
      EXPECT_EQ(settings->max_slots.value, static_cast<std::uint32_t>(owners));
      EXPECT_EQ(settings->prefill_chunk.value, 256U);
      // The real source, state and paid backend admission still decide
      // whether this requested geometry is executable.
    }
    entry.overrides["max_slots"] = std::int64_t{17};
    EXPECT_FALSE(ResolveSettings(entry, facts, nullptr, false));
    entry = Model("gemma");
    Calibration calibration;
    calibration.max_slots = 4;
    EXPECT_TRUE(ResolveSettings(entry, facts, &calibration, false));
    entry.overrides["context"] = std::int64_t{262145};
    EXPECT_FALSE(ResolveSettings(entry, facts, nullptr, false));
  }
}

TEST(ModelSettings, PinnedGemma23ServingEnvelopesKeepTheFourOwnerDefault) {
  for (const auto& [architecture, name, context] :
       {std::tuple{"gemma2", "gemma-2-2b-it-q8_0", 8192U},
        std::tuple{"gemma3", "gemma-3-4b-it-qat-q4_0", 4096U}}) {
    auto facts = Facts(architecture);
    facts.approval = llmp::runtime::FindApprovalNamed(name);
    ASSERT_NE(facts.approval, nullptr);
    auto entry = Model("gemma");
    entry.overrides["context"] = std::int64_t{context};
    auto settings = ResolveSettings(entry, facts, nullptr, false);
    ASSERT_TRUE(settings);
    EXPECT_EQ(settings->max_slots.value, 4U);
    // This is a serving geometry/default. Numerical transformation authority
    // remains a separate immutable per-source/component/shape qualification.
    EXPECT_FALSE(settings->speculation.value);
  }
}

TEST(ModelSettings, OptionalRecipesCannotVetoAStoredDrafter) {
  for (const auto* architecture : {"gemma2", "gemma3", "unregistered-target"}) {
    auto facts = Facts(architecture, "unregistered-drafter");
    facts.context_ceiling = 4096;
    facts.target_spec = llmp::base::Sha256Digest{};
    facts.composed_spec = llmp::base::Sha256Digest{};
    const auto drafted = ResolveSettings(Model("generic", true), facts, nullptr, false);
    ASSERT_TRUE(drafted) << drafted.error();
    EXPECT_TRUE(drafted->speculation.value);
    facts.composed_spec.reset();
    facts.composed_spec_error = "invalid stored drafter";
    EXPECT_FALSE(ResolveSettings(Model("generic", true), facts, nullptr, false));
    auto entry = Model("generic");
    entry.overrides["speculation"] = true;
    const auto plain = ResolveSettings(entry, facts, nullptr, false);
    ASSERT_TRUE(plain) << plain.error();
    EXPECT_FALSE(plain->speculation.value);
  }
}

TEST(ModelSettings, ReportsTheSelectedDraftHeadsPhysicalLimit) {
  ArtifactFacts facts = Facts("qwen4exp", "qwen4exp-mtp");
  facts.drafter_selected_rows = 47172;
  const ModelSettings fallback = Resolved(Model("q", true), facts);
  EXPECT_EQ(fallback.draft_vocab.value, 47172U);
  EXPECT_EQ(fallback.draft_vocab.source, SettingSource::kDerived);
  EXPECT_THAT(fallback.draft_vocab.basis, HasSubstr("65536 requested"));
  EXPECT_THAT(fallback.draft_vocab.basis, HasSubstr("47172 rows"));
  for (const std::uint32_t asked : {0U, 32768U, 65536U}) {
    ModelEntry owner = Model("q", true);
    owner.overrides["draft_vocab"] = static_cast<std::int64_t>(asked);
    const ModelSettings result = Resolved(owner, facts);
    EXPECT_EQ(result.draft_vocab.value, asked == 32768U ? asked : 47172U);
    EXPECT_EQ(result.draft_vocab.source, SettingSource::kOverride);
  }
  ModelEntry oversized = Model("q", true);
  oversized.overrides["draft_vocab"] = std::int64_t{4194304};
  EXPECT_EQ(Resolved(oversized, facts).draft_vocab.value, 47172U);
  const ModelSettings prefix = Resolved(Model("q", true), Facts("qwen4exp", "qwen4exp-mtp"));
  EXPECT_EQ(prefix.draft_vocab.value, 65536U);
  EXPECT_EQ(prefix.draft_vocab.source, SettingSource::kFallback);
  const ModelSettings plain = Resolved(Model("q", true), facts, nullptr, true);
  EXPECT_FALSE(plain.speculation.value);
  EXPECT_EQ(plain.draft_vocab.value, 65536U);
}

TEST(ModelSettings, ZeroDraftVocabularyWithoutSelectedHeadBindsTheFullVocabulary) {
  auto facts = Facts("qwen4exp", "qwen4exp_mtp");
  facts.composed_draft_vocab = facts.vocabulary;
  facts.target_spec = llmp::base::Sha256Digest{};
  facts.composed_spec = llmp::base::Sha256Digest{};
  auto entry = Model("full", true);
  entry.overrides["draft_vocab"] = std::int64_t{0};
  const auto settings = ResolveSettings(entry, facts, nullptr, false);
  ASSERT_TRUE(settings) << (settings ? "" : settings.error());
  EXPECT_EQ(settings->draft_vocab.value, 0U);
  EXPECT_EQ(settings->bound_spec, facts.composed_spec);
}

TEST(ModelSettings, ActualPrefixHeadBoundsDefaultAndExplicitVocabulary) {
  for (const auto requested : {0U, 32U, 64U, 128U}) {
    auto facts = Facts("unknown-target", "unknown-drafter");
    facts.drafter_mtp = true;
    facts.context_ceiling = 4096;
    facts.vocabulary = 128;
    facts.drafter_head_rows = 64;
    facts.composed_draft_vocab = requested == 0 ? 64U : std::min(requested, 64U);
    facts.target_spec = llmp::base::Sha256Digest{};
    facts.composed_spec = llmp::base::Sha256Digest{};
    auto entry = Model("prefix", true);
    entry.overrides["draft_vocab"] = std::int64_t{requested};
    const auto settings = ResolveSettings(entry, facts, nullptr, false);
    ASSERT_TRUE(settings) << settings.error();
    EXPECT_EQ(settings->draft_vocab.value, *facts.composed_draft_vocab);
    facts.composed_draft_vocab = 17;
    EXPECT_FALSE(ResolveSettings(entry, facts, nullptr, false));
  }
}

TEST(ModelSettings, DraftDepthUsesStoredSourceAndRepresentableExecutionBounds) {
  auto facts = Facts("unknown-target", "unknown-drafter");
  facts.context_ceiling = 4096;
  facts.target_spec = llmp::base::Sha256Digest{};
  facts.composed_spec = llmp::base::Sha256Digest{};
  auto entry = Model("generic", true);
  for (const auto depth : {17U, UINT32_MAX - 1}) {
    entry.overrides["draft_rows"] = std::int64_t{depth};
    const auto settings = ResolveSettings(entry, facts, nullptr, false);
    ASSERT_TRUE(settings) << settings.error();
    EXPECT_EQ(settings->draft_rows.value, depth);
  }
  entry.overrides["draft_rows"] = std::int64_t{UINT32_MAX};
  EXPECT_FALSE(ResolveSettings(entry, facts, nullptr, false));
  entry.overrides["draft_rows"] = std::int64_t{17};
  facts.drafter_block = 16;
  EXPECT_FALSE(ResolveSettings(entry, facts, nullptr, false));
}

TEST(ModelSettings, DraftVocabularyIdentityRequiresFactsForTheSameOverride) {
  auto facts = Facts("qwen4exp", "qwen4exp_mtp");
  facts.drafter_selected_rows = 47172;
  facts.composed_draft_vocab = 32768;
  facts.target_spec = llmp::base::Sha256Digest{};
  facts.composed_spec = llmp::base::Sha256Digest{};
  auto entry = Model("narrow", true);
  entry.overrides["draft_vocab"] = std::int64_t{32768};
  const auto resolved = ResolveSettings(entry, facts, nullptr, false);
  ASSERT_TRUE(resolved) << (resolved ? "" : resolved.error());
  EXPECT_EQ(resolved->draft_vocab.value, 32768U);
  entry.overrides["draft_vocab"] = std::int64_t{16384};
  EXPECT_FALSE(ResolveSettings(entry, facts, nullptr, false));
  EXPECT_TRUE(ResolveSettings(entry, facts, nullptr, true));
}

TEST(ModelSettings, OwnerCanDisableConcurrentWaveLanes) {
  ModelEntry entry = Model("q");
  entry.overrides["wave_lanes"] = false;
  const ModelSettings q = Resolved(entry, Facts("qwen4exp"));
  EXPECT_FALSE(q.wave_lanes.value);
  EXPECT_EQ(q.wave_lanes.source, SettingSource::kOverride);
  EXPECT_THAT(q.Summary(), HasSubstr("wave_lanes=false (override)"));
}

// An approved checkpoint's envelope is its default (D-119), not a bound: a
// configuration that names no slots resolves within it, the basis saying
// so; one that names more owners or a wider context than it qualified is
// served, with a warning, never refused at setup for it. A pinned
// checkpoint is no more restricted than an unpinned re-quant.
TEST(ModelSettings, AnApprovedEnvelopeIsADefaultAnOverridePassesWithAWarning) {
  const llmp::runtime::Approval approval{
      .name = "test", .label = "Test", .max_context = 131072, .max_slots = 2};
  const llmp::runtime::Approval wide{.name = "wide",
                                     .label = "Wide",
                                     .max_context = 262144,
                                     .max_slots = 8,
                                     .wide_context = 65536};
  auto facts = Facts("qwen4exp");
  facts.approval = &approval;
  const ModelSettings defaults = Resolved(Model("q"), facts);
  EXPECT_EQ(defaults.max_slots.value, 2U);
  EXPECT_EQ(defaults.max_slots.source, SettingSource::kDerived);
  EXPECT_THAT(defaults.max_slots.basis, HasSubstr("approved Test envelope"));
  EXPECT_LE(defaults.context.value, 131072U);
  EXPECT_TRUE(defaults.warnings.empty());
  auto entry = Model("q");
  entry.overrides["max_slots"] = std::int64_t{6};
  entry.overrides["context"] = std::int64_t{200000};
  const ModelSettings past = Resolved(entry, facts);
  EXPECT_EQ(past.max_slots.value, 6U);
  EXPECT_EQ(past.context.value, 200000U);
  ASSERT_EQ(past.warnings.size(), 2U);
  EXPECT_THAT(past.warnings[0], HasSubstr("context 200000 is past the approved Test envelope"));
  EXPECT_THAT(past.warnings[1], HasSubstr("max_slots 6 is past the approved Test envelope"));
  // The same without an approval: no warning, the same values.
  facts.approval = nullptr;
  const ModelSettings unpinned = Resolved(entry, facts);
  EXPECT_EQ(unpinned.max_slots.value, 6U);
  EXPECT_TRUE(unpinned.warnings.empty());
  // Above a wide context, one owner by default.
  facts.approval = &wide;
  entry.overrides.erase("max_slots");
  entry.overrides["context"] = std::int64_t{100000};
  const ModelSettings alone = Resolved(entry, facts);
  EXPECT_EQ(alone.max_slots.value, 1U);
  EXPECT_THAT(alone.max_slots.basis, HasSubstr("one owner above 65536"));
  EXPECT_TRUE(alone.warnings.empty());
}

TEST(ModelSettings, UnpinnedGemma3ClampsMaxSlotsToOneAboveWideContext) {
  auto facts = Facts("gemma3");
  facts.approval = nullptr;
  auto entry = Model("gemma3-unpinned");
  entry.overrides["context"] = std::int64_t{8192};
  const ModelSettings settings = Resolved(entry, facts);
  EXPECT_EQ(settings.context.value, 8192U);
  EXPECT_EQ(settings.max_slots.value, 1U);
  EXPECT_EQ(settings.max_slots.source, SettingSource::kDerived);
  EXPECT_THAT(settings.max_slots.basis, HasSubstr("Gemma3 recipe's one owner above 4096 tokens"));
}

TEST(ModelSettings, GemmaDefaultsKeepAnUncalibratedFourOwnerPreference) {
  auto entry = Model("gemma");
  auto facts = Facts("gemma4");
  facts.trained_context = 262144;
  facts.trained_context_from = "gemma4.context_length";
  const auto s = Resolved(entry, facts);
  EXPECT_EQ(s.context.value, 262144U);
  EXPECT_EQ(s.max_slots.value, 4U);
  EXPECT_EQ(s.prefill_chunk.value, 128U);
  EXPECT_THAT(s.max_slots.basis, HasSubstr("uncalibrated"));
  EXPECT_THAT(s.prefill_chunk.basis, HasSubstr("uncalibrated"));
  EXPECT_THAT(s.recompute_ms_per_token.basis, HasSubstr("uncalibrated"));
  EXPECT_FALSE(s.speculation.value);
  for (const auto owners : {1, 2, 4, 8, 16}) {
    entry.overrides["max_slots"] = std::int64_t{owners};
    const auto chosen = Resolved(entry, facts);
    EXPECT_EQ(chosen.max_slots.value, static_cast<std::uint32_t>(owners));
  }
  entry.overrides["max_slots"] = std::int64_t{17};
  EXPECT_FALSE(ResolveSettings(entry, facts, nullptr, false));
  entry.overrides["max_slots"] = std::int64_t{12};
  entry.overrides["prefill_chunk"] = std::int64_t{4};
  EXPECT_TRUE(ResolveSettings(entry, facts, nullptr, false));
  entry.overrides["prefill_chunk"] = std::int64_t{4096};
  EXPECT_EQ(Resolved(entry, facts).prefill_chunk.value, 4096U);
  entry.overrides["context"] = std::int64_t{262145};
  EXPECT_FALSE(ResolveSettings(entry, facts, nullptr, false));
  // Gemma 4 drafts with its assistant; without one, speculation is ignored.
  const auto drafted = Resolved(Model("gemma", true), Facts("gemma4", "gemma4-assistant"));
  EXPECT_TRUE(drafted.speculation.value);
  EXPECT_THAT(drafted.speculation.basis, HasSubstr("gemma4-assistant"));
  entry = Model("gemma");
  entry.overrides["speculation"] = true;
  EXPECT_FALSE(Resolved(entry, facts).speculation.value);
  // With no configured companion, speculation stays disabled for every target.
  for (const char* arch : {"gemma2", "gemma3"})
    EXPECT_FALSE(Resolved(entry, Facts(arch)).speculation.value) << arch;
}

TEST(ModelSettings, Gemma31ProductionDefaultIsBoundedAndProfileSpecific) {
  auto entry = Model("gemma");
  entry.overrides["context"] = std::int64_t{8192};
  entry.overrides["prefill_chunk"] = std::int64_t{1024};
  entry.overrides["max_slots"] = std::int64_t{4};
  auto facts = Facts("gemma4");
  facts.approval = llmp::runtime::FindApprovalNamed("gemma-4-31b-it-ud-q4_k_xl");
  auto candidate = ResolveSettings(entry, facts, nullptr, true);
  ASSERT_TRUE(candidate);
  EXPECT_EQ(candidate->prefill_chunk.value, 1024U);
  EXPECT_EQ(candidate->max_slots.value, 4U);
  EXPECT_FALSE(candidate->speculation.value);
  entry.overrides["context"] = std::int64_t{8193};
  auto larger_context = ResolveSettings(entry, facts, nullptr, true);
  ASSERT_TRUE(larger_context);
  EXPECT_EQ(larger_context->prefill_chunk.value, 1024U);
  entry.overrides["context"] = std::int64_t{8192};
  entry.overrides["max_slots"] = std::int64_t{5};
  auto larger_cohort = ResolveSettings(entry, facts, nullptr, true);
  ASSERT_TRUE(larger_cohort);
  EXPECT_EQ(larger_cohort->prefill_chunk.value, 1024U);
  entry.overrides["max_slots"] = std::int64_t{4};
  facts.approval = llmp::runtime::FindApprovalNamed("gemma-4-26b-a4b-it-ud-q4_k_m");
  auto profile26 = ResolveSettings(entry, facts, nullptr, true);
  ASSERT_TRUE(profile26);
  EXPECT_EQ(profile26->prefill_chunk.value, 1024U);
  facts.approval = nullptr;
  auto unknown = ResolveSettings(entry, facts, nullptr, true);
  ASSERT_TRUE(unknown);
  EXPECT_EQ(unknown->prefill_chunk.value, 1024U);
}

// A default configuration resolves within its approval: Gemma 4's joined
// recipes stop at 8,192 tokens and four owners (its owners-alone route,
// to 262,144 tokens and twelve owners, is an explicit opt-in), so its
// default context is theirs (and with its assistant within the assistant's
// too), and its recipe's qualified chunk follows.
TEST(ModelSettings, Gemma4DefaultsResolveWithinTheirApprovals) {
  for (const auto& [target, assistant, chunk] :
       {std::tuple{"gemma-4-26b-a4b-it-ud-q4_k_m", "mtp-gemma-4-26b-a4b-it", 1024U},
        std::tuple{"gemma-4-31b-it-ud-q4_k_xl", "mtp-gemma-4-31b-it", 256U}}) {
    SCOPED_TRACE(target);
    for (const bool drafted : {false, true}) {
      auto facts = drafted ? Facts("gemma4", "gemma4-assistant") : Facts("gemma4");
      facts.approval = llmp::runtime::FindApprovalNamed(target);
      ASSERT_NE(facts.approval, nullptr);
      if (drafted) facts.drafter_approval = llmp::runtime::FindApprovalNamed(assistant);
      const auto settings = ResolveSettings(Model("gemma", drafted), facts, nullptr, false);
      ASSERT_TRUE(settings) << settings.error();
      EXPECT_EQ(settings->context.value, 8192U);
      EXPECT_EQ(settings->context.value, llmp::runtime::ApprovedDefaultContext(*facts.approval));
      EXPECT_EQ(settings->context.source, SettingSource::kDerived);
      EXPECT_THAT(settings->context.basis, HasSubstr("approved"));
      EXPECT_LE(settings->max_slots.value, facts.approval->max_slots);
      EXPECT_EQ(settings->prefill_chunk.value, chunk);
      if (drafted) {
        EXPECT_TRUE(settings->speculation.value);
        EXPECT_LE(settings->context.value, facts.drafter_approval->max_context);
        EXPECT_LE(settings->max_slots.value, facts.drafter_approval->max_slots);
      }
    }
  }
  // An override still names its own context (the owners-alone route admits
  // it; the approval refuses one past its envelope).
  auto facts = Facts("gemma4");
  facts.approval = llmp::runtime::FindApprovalNamed("gemma-4-31b-it-ud-q4_k_xl");
  auto entry = Model("gemma");
  entry.overrides["context"] = std::int64_t{16384};
  const auto wide = ResolveSettings(entry, facts, nullptr, false);
  ASSERT_TRUE(wide);
  EXPECT_EQ(wide->context.value, 16384U);
}

TEST(ModelSettings, Gemma26ProductionUses1024WithinBoundsAndPreservesSmallerOverride) {
  auto entry = Model("gemma");
  auto facts = Facts("gemma4");
  facts.approval = llmp::runtime::FindApprovalNamed("gemma-4-26b-a4b-it-ud-q4_k_m");
  // The default context is the approved envelope's: its qualified chunk.
  auto defaults = ResolveSettings(entry, facts, nullptr, true);
  ASSERT_TRUE(defaults);
  EXPECT_EQ(defaults->context.value, 8192U);
  EXPECT_EQ(defaults->prefill_chunk.value, 1024U);
  // A wider context than the envelope: the prior recipe.
  entry.overrides["context"] = std::int64_t{16384};
  auto unbounded = ResolveSettings(entry, facts, nullptr, true);
  ASSERT_TRUE(unbounded);
  EXPECT_EQ(unbounded->prefill_chunk.value, 128U);
  entry.overrides["context"] = std::int64_t{8192};
  auto candidate = ResolveSettings(entry, facts, nullptr, true);
  ASSERT_TRUE(candidate);
  EXPECT_EQ(candidate->prefill_chunk.value, 1024U);
  entry.overrides["max_slots"] = std::int64_t{4};
  candidate = ResolveSettings(entry, facts, nullptr, true);
  ASSERT_TRUE(candidate);
  entry.overrides["max_slots"] = std::int64_t{5};
  auto larger_cohort = ResolveSettings(entry, facts, nullptr, true);
  ASSERT_TRUE(larger_cohort);
  EXPECT_EQ(larger_cohort->prefill_chunk.value, 128U);
  entry.overrides["max_slots"] = std::int64_t{4};
  entry.overrides["context"] = std::int64_t{8193};
  auto larger_context = ResolveSettings(entry, facts, nullptr, true);
  ASSERT_TRUE(larger_context);
  entry.overrides["context"] = std::int64_t{8192};
  entry.overrides["prefill_chunk"] = std::int64_t{2048};
  candidate = ResolveSettings(entry, facts, nullptr, true);
  ASSERT_TRUE(candidate);
  EXPECT_EQ(candidate->prefill_chunk.value, 2048U);
  entry.overrides["prefill_chunk"] = std::int64_t{64};
  candidate = ResolveSettings(entry, facts, nullptr, true);
  ASSERT_TRUE(candidate);
  EXPECT_EQ(candidate->prefill_chunk.value, 64U);
}

TEST(ModelSettings, Gemma31ProductionUses256WithinBoundsAndPreservesSmallerOverride) {
  auto entry = Model("gemma");
  auto facts = Facts("gemma4");
  facts.approval = llmp::runtime::FindApprovalNamed("gemma-4-31b-it-ud-q4_k_xl");
  auto candidate = ResolveSettings(entry, facts, nullptr, true);
  ASSERT_TRUE(candidate);
  EXPECT_EQ(candidate->context.value, 8192U);  // the approved envelope's
  EXPECT_EQ(candidate->prefill_chunk.value, 256U);
  EXPECT_EQ(candidate->max_slots.value, 4U);
  entry.overrides["context"] = std::int64_t{16384};
  candidate = ResolveSettings(entry, facts, nullptr, true);
  ASSERT_TRUE(candidate);
  EXPECT_EQ(candidate->prefill_chunk.value, 128U);
  entry.overrides["context"] = std::int64_t{8192};
  candidate = ResolveSettings(entry, facts, nullptr, true);
  ASSERT_TRUE(candidate);
  EXPECT_EQ(candidate->prefill_chunk.value, 256U);
  entry.overrides["prefill_chunk"] = std::int64_t{16};
  candidate = ResolveSettings(entry, facts, nullptr, true);
  ASSERT_TRUE(candidate);
  EXPECT_EQ(candidate->prefill_chunk.value, 16U);
}

// Derived from the artifact: the trained context, the checkpoint's
// sampling defaults, the drafter's block.
TEST(ModelSettings, DerivesFromTheArtifact) {
  ArtifactFacts facts = Facts("deepseek4", "dflash");
  facts.trained_context = 32768;
  facts.trained_context_from = "deepseek4.context_length";
  facts.temperature = 0.6;
  facts.top_p = 0.95;
  facts.top_k = 20;
  facts.sampling_from = "generation_config.json";
  facts.drafter_block = 2;
  const ModelSettings s = Resolved(Model("ds", true), facts);
  EXPECT_EQ(s.context.value, 32768U);
  EXPECT_EQ(s.context.source, SettingSource::kDerived);
  EXPECT_THAT(s.context.basis, HasSubstr("deepseek4.context_length"));
  EXPECT_TRUE(s.speculation.value);
  EXPECT_EQ(s.speculation.source, SettingSource::kDerived);
  EXPECT_EQ(s.temperature.value, 0.6);
  EXPECT_EQ(s.temperature.source, SettingSource::kDerived);
  EXPECT_EQ(s.top_p.value, 0.95);
  EXPECT_EQ(s.top_k.value, 20U);
  EXPECT_EQ(s.min_p.source, SettingSource::kFallback);
  EXPECT_EQ(s.draft_rows.value, 2U);
  EXPECT_EQ(s.draft_rows.source, SettingSource::kDerived);

  // A trained context past the runner's: the runner's ceiling, and the
  // default within it (DeepSeek V4 Flash's GGUF says 1,048,576).
  facts.trained_context = 1048576;
  const ModelSettings wide = Resolved(Model("ds", true), facts);
  EXPECT_EQ(wide.context.value, 262144U);
  EXPECT_EQ(wide.context.source, SettingSource::kFallback);
  facts.trained_context = 4194304;
  const auto wider = ResolveSettings(Model("ds"), facts, nullptr, false);
  ASSERT_TRUE(wider.has_value());
  EXPECT_EQ(wider.value_or(ModelSettings{}).context.value, 262144U);
}

// A v0 artifact is refused at open time by the v1-only reader (D-118, U2),
// and a v1 artifact without a stored spec is refused by ArtifactFactsOf.
TEST(ModelSettings, ArtifactFactsRefuseV0Artifacts) {
  const std::filesystem::path scratch =
      std::filesystem::path(testing::TempDir()) / "model-settings-v0";
  std::error_code error;
  std::filesystem::remove_all(scratch, error);
  std::filesystem::create_directories(scratch, error);

  llmp::test::V1Plan plan;
  plan.architecture = "toy";
  plan.groups = {
      llmp::test::V1Group{.kind = llmp::artifact::GroupKind::kTable,
                          .resources = {{.name = "token_embd.weight",
                                         .format = {.encoding = llmp::format::Encoding::kF32,
                                                    .layout = llmp::format::Layout::kDense,
                                                    .shape = {.ne = {64, 64, 1, 1}, .rank = 2}}}}}};
  auto v0_rendered = llmp::test::Render(plan);
  v0_rendered.manifest["format"] = llmp::test::Json("jitllm-artifact");
  v0_rendered.manifest["format_version"] = llmp::test::Json(std::int64_t{0});
  const auto v0 = llmp::artifact::Artifact::Open(llmp::test::Publish(scratch / "v0", v0_rendered));
  ASSERT_FALSE(v0.has_value());
  EXPECT_EQ(v0.error().rule, llmp::artifact::Rule::kFormat);

  const auto v1 =
      llmp::artifact::Artifact::Open(llmp::test::Publish(scratch / "v1", llmp::test::Render(plan)));
  ASSERT_TRUE(v1.has_value()) << (v1 ? "" : v1.error().ToString());
  const auto alone = llmp::runtime::ArtifactFactsOf(*v1, nullptr);
  ASSERT_FALSE(alone.has_value());
}

// The facts a stored v1 spec gives: stage.context, stage.sampling
// (temperature, top_p, top_k, min_p, including greedy temperature 0.0),
// and a standalone two-stage drafter spec.json composed with its target
// without reading checkpoint metadata sidecars.
TEST(ModelSettings, ReadsFactsFromMetadata) {
  const std::filesystem::path scratch =
      std::filesystem::path(testing::TempDir()) / "model-settings-stored-spec";
  std::error_code error;
  std::filesystem::remove_all(scratch, error);
  std::filesystem::create_directories(scratch, error);

  const auto make_stage = [](std::string name, llmp::spec::StageRole role, std::uint32_t artifact,
                             std::uint32_t context, llmp::spec::SamplingDefaults sampling) {
    llmp::spec::Stage s{
        .name = std::move(name),
        .role = role,
        .artifact = artifact,
        .context = context,
        .sampling = sampling,
    };
    s.embedding =
        llmp::spec::ComponentRef{.kind = "embedding.token",
                                 .version = 1,
                                 .params = {{"scale", 1.0}},
                                 .weights = {{.role = "table", .tensor = "token_embd.weight"}}};
    s.stream.width = 32;
    s.layers.push_back(llmp::spec::Layer{
        .components = {llmp::spec::ComponentRef{
            .kind = "norm.rms",
            .version = 1,
            .params = {{"eps", 0.00001}},
            .weights = {{.role = "weight", .tensor = "blk.0.attn_norm.weight"}}}}});
    s.heads.output =
        llmp::spec::ComponentRef{.kind = "head.linear",
                                 .version = 1,
                                 .params = {{"rows", std::int64_t{64}}},
                                 .weights = {{.role = "weight", .tensor = "output.weight"}}};
    return s;
  };
  const auto publish_v1 = [&](const std::filesystem::path& root, std::string arch,
                              const llmp::spec::ModelSpec& spec, std::string container = "gguf",
                              std::uint32_t head_rows = 64) {
    llmp::test::V1Plan plan;
    plan.container = std::move(container);
    plan.architecture = std::move(arch);
    plan.spec = llmp::spec::ToJson(spec);
    plan.groups = {llmp::test::V1Group{
        .kind = llmp::artifact::GroupKind::kTable,
        .resources = {{.name = "token_embd.weight",
                       .aliases = {"output.weight"},
                       .format = {.encoding = llmp::format::Encoding::kF32,
                                  .layout = llmp::format::Layout::kDense,
                                  .shape = {.ne = {32, 64, 1, 1}, .rank = 2}},
                       .rows = true},
                      {.name = "blk.0.attn_norm.weight",
                       .format = {.encoding = llmp::format::Encoding::kF32,
                                  .layout = llmp::format::Layout::kDense,
                                  .shape = {.ne = {32, 1, 1, 1}, .rank = 1}}}}}};
    if (spec.stages.size() == 2)
      plan.groups.front().resources.push_back(
          {.name = "projection.weight",
           .format = {.encoding = llmp::format::Encoding::kF32,
                      .layout = llmp::format::Layout::kDense,
                      .shape = {.ne = {64, 32, 1, 1}, .rank = 2}}});
    if (head_rows != 64) {
      plan.groups.front().resources.front().aliases.clear();
      plan.groups.front().resources.push_back(
          {.name = "output.weight",
           .format = {.encoding = llmp::format::Encoding::kF32,
                      .layout = llmp::format::Layout::kDense,
                      .shape = {.ne = {32, head_rows, 1, 1}, .rank = 2}}});
    }
    return llmp::test::Publish(root, llmp::test::Render(plan));
  };

  llmp::spec::ModelSpec ds_spec{
      .pipeline = llmp::spec::PipelineKind::kLlm,
      .stages = {make_stage("target", llmp::spec::StageRole::kTarget, 0, 1048576,
                            {.temperature = 0.6, .top_p = 0.95, .top_k = 20, .min_p = 0.05})}};
  llmp::spec::Canonicalize(ds_spec);
  const auto ds_path = publish_v1(scratch / "ds", "deepseek4", ds_spec);
  const auto ds_art = llmp::artifact::Artifact::Open(ds_path);
  ASSERT_TRUE(ds_art.has_value()) << (ds_art ? "" : ds_art.error().ToString());
  const auto ds = llmp::runtime::ArtifactFactsOf(*ds_art, nullptr);
  ASSERT_TRUE(ds.has_value()) << (ds ? "" : ds.error());
  EXPECT_EQ(ds->trained_context, 1048576U);
  EXPECT_EQ(ds->trained_context_from, "the admitted stored spec");
  EXPECT_EQ(ds->temperature, 0.6);
  EXPECT_EQ(ds->top_p, 0.95);
  EXPECT_EQ(ds->top_k, 20U);
  EXPECT_EQ(ds->min_p, 0.05);
  EXPECT_EQ(ds->sampling_from, "the admitted stored spec");

  // The same executable stored components do not need an import registry entry.
  const auto generic_path = publish_v1(scratch / "generic", "unregistered_decoder", ds_spec);
  const auto generic_art = llmp::artifact::Artifact::Open(generic_path);
  ASSERT_TRUE(generic_art);
  const auto generic = llmp::runtime::ArtifactFactsOf(*generic_art, nullptr);
  ASSERT_TRUE(generic) << generic.error();
  EXPECT_EQ(generic->approval, nullptr);
  EXPECT_EQ(generic->trained_context, ds->trained_context);
  EXPECT_EQ(generic->temperature, ds->temperature);
  const auto generic_settings = Resolved(Model("generic"), *generic);
  EXPECT_EQ(generic_settings.prefill_chunk.value, llmp::spec::kDefaultServingRecipe.prefill_rows);

  const ModelSettings sampled = Resolved(Model("ds"), *ds);
  EXPECT_EQ(sampled.temperature.value, 0.6);
  EXPECT_EQ(sampled.temperature.source, SettingSource::kDerived);
  EXPECT_EQ(sampled.top_p.value, 0.95);
  EXPECT_EQ(sampled.top_p.source, SettingSource::kDerived);
  EXPECT_EQ(sampled.top_k.value, 20U);
  EXPECT_EQ(sampled.min_p.value, 0.05);
  EXPECT_THAT(sampled.Summary(), HasSubstr("top_p=0.95 (derived) top_k=20 (derived)"));

  // Greedy sampling (temperature 0.0) and a derived context below the
  // architecture's default.
  llmp::spec::ModelSpec greedy_spec{
      .pipeline = llmp::spec::PipelineKind::kLlm,
      .stages = {make_stage("target", llmp::spec::StageRole::kTarget, 0, 40960,
                            {.temperature = 0.0, .top_p = 0.9})}};
  llmp::spec::Canonicalize(greedy_spec);
  const auto greedy_art =
      llmp::artifact::Artifact::Open(publish_v1(scratch / "greedy", "deepseek4", greedy_spec));
  ASSERT_TRUE(greedy_art.has_value()) << (greedy_art ? "" : greedy_art.error().ToString());
  const auto greedy = llmp::runtime::ArtifactFactsOf(*greedy_art, nullptr);
  ASSERT_TRUE(greedy.has_value()) << (greedy ? "" : greedy.error());
  EXPECT_EQ(greedy->temperature, 0.0);
  EXPECT_EQ(greedy->top_p, 0.9);
  EXPECT_FALSE(greedy->top_k.has_value());
  const ModelSettings small = Resolved(Model("ds"), *greedy);
  EXPECT_EQ(small.context.value, 40960U);
  EXPECT_EQ(small.context.source, SettingSource::kDerived);
  EXPECT_EQ(small.temperature.value, 0.0);
  EXPECT_EQ(small.temperature.source, SettingSource::kDerived);

  // A standalone two-stage drafter spec.json composed with the target via
  // ReadArtifactFacts, returning the opened artifacts and admitted specs.
  llmp::spec::Stage drafter_stage =
      make_stage("drafter", llmp::spec::StageRole::kDrafter, 1, 1048576, {});
  drafter_stage.drafting = llmp::spec::MtpDraft{};
  drafter_stage.embedding->kind = "embedding.projected";
  drafter_stage.embedding->params = {
      {"scale", 1.0}, {"feature", std::int64_t{32}}, {"tap", std::string("features")}};
  drafter_stage.embedding->weights.push_back({.role = "projection", .tensor = "projection.weight"});
  drafter_stage.reads = {
      {.stage = "target", .kind = llmp::spec::ReadKind::kTap, .name = "features"}};
  llmp::spec::ModelSpec drafter_spec{.pipeline = llmp::spec::PipelineKind::kLlm,
                                     .stages = {ds_spec.stages.front(), std::move(drafter_stage)}};
  drafter_spec.stages.front().taps = {
      {.name = "features", .layer = 0, .point = llmp::spec::TapPoint::kLayerOut}};
  llmp::spec::Canonicalize(drafter_spec);
  const auto store_dir = scratch / "store";
  const auto pub_target = publish_v1(store_dir, "deepseek4", ds_spec);
  const auto pub_drafter = publish_v1(store_dir, "dflash", drafter_spec);

  ModelEntry composed_entry;
  composed_entry.name = "ds";
  composed_entry.artifact = pub_target.filename().string();
  composed_entry.drafter = pub_drafter.filename().string();
  llmp::runtime::OpenedModelArtifacts opened;
  const auto composed_facts = llmp::runtime::ReadArtifactFacts(composed_entry, store_dir, &opened);
  ASSERT_TRUE(composed_facts.has_value()) << (composed_facts ? "" : composed_facts.error());
  EXPECT_TRUE(composed_facts->target_spec.has_value());
  EXPECT_TRUE(composed_facts->composed_spec.has_value());
  EXPECT_TRUE(opened.target.has_value());
  EXPECT_TRUE(opened.drafter.has_value());
  EXPECT_TRUE(opened.target_admission.has_value());
  ASSERT_TRUE(opened.drafter_admission.has_value());
  EXPECT_TRUE(opened.drafter_admission->model.has_value());
  ASSERT_TRUE(opened.composition);
  const auto shared_composition = opened.composition;
  EXPECT_EQ(shared_composition->model().stages.size(), 2U);
  const auto original_semantic = shared_composition->numerical_semantic();
  auto changed_copy = shared_composition->model();
  changed_copy.stages.front().context = 64;
  EXPECT_NE(changed_copy, shared_composition->model());
  EXPECT_EQ(shared_composition->numerical_semantic(), original_semantic);
  EXPECT_FALSE(
      shared_composition->CheckBinding(*opened.target, *opened.drafter, changed_copy, 64, 0));
  EXPECT_FALSE(shared_composition->CheckBinding(*greedy_art, *opened.drafter,
                                                opened.target_admission->model, 64, 0));
  EXPECT_TRUE(shared_composition->CheckBinding(*opened.target, *opened.drafter,
                                               opened.target_admission->model, 64, 0));

  // Source architecture spellings survive publication, while both native and
  // MLX MTP recipes name the canonical target. Stored drafter composition and
  // calibration must use that resolved identity, with structural checks intact.
  for (const std::string draft_container : {"safetensors", "mlx"}) {
    const auto mtp_path = publish_v1(store_dir, "qwen4exp-mtp", drafter_spec, draft_container);
    const auto mtp = llmp::artifact::Artifact::Open(mtp_path);
    ASSERT_TRUE(mtp) << mtp.error().ToString();
    for (const std::string target_container : {"safetensors", "mlx"}) {
      for (const std::string architecture : {"qwen4exp", "qwen3_8_flash_next", "qwen4_exp"}) {
        SCOPED_TRACE(draft_container + ":" + target_container + ":" + architecture);
        ModelEntry entry = Model("mtp", true);
        entry.overrides["draft_vocab"] = std::int64_t{64};
        entry.artifact =
            publish_v1(store_dir, architecture, ds_spec, target_container).filename().string();
        entry.drafter = mtp_path.filename().string();
        llmp::runtime::OpenedModelArtifacts retained;
        const auto facts = llmp::runtime::ReadArtifactFacts(entry, store_dir, &retained);
        ASSERT_TRUE(facts) << facts.error();
        EXPECT_EQ(facts->architecture, architecture);
        ASSERT_TRUE(facts->composed_spec) << facts->composed_spec_error;
        EXPECT_TRUE(facts->composed_spec_error.empty());
        ASSERT_TRUE(retained.drafter_admission);
        ASSERT_TRUE(retained.drafter_admission->model);
        Calibration calibration;
        calibration.prefill_chunk = 64;
        const auto settings = ResolveSettings(entry, *facts, &calibration, false);
        ASSERT_TRUE(settings) << settings.error();
        EXPECT_EQ(settings->bound_spec, facts->composed_spec);
        ASSERT_TRUE(retained.composition);
        EXPECT_EQ(retained.composition->vocabulary(), 64U);
        EXPECT_TRUE(retained.composition->CheckBinding(*retained.target, *retained.drafter,
                                                       retained.target_admission->model, 64, 0));
        EXPECT_FALSE(retained.composition->CheckBinding(*retained.target, *retained.drafter,
                                                        retained.target_admission->model, 64, 32));
        EXPECT_TRUE(settings->speculation.value);
      }
    }
    const auto another_name = llmp::runtime::ArtifactFactsOf(*ds_art, &*mtp);
    ASSERT_TRUE(another_name) << another_name.error();
    EXPECT_TRUE(another_name->composed_spec) << another_name->composed_spec_error;
  }
  // Physical prefix width, including the default request, comes from the
  // opened output tensor rather than the larger target vocabulary.
  auto prefix_spec = drafter_spec;
  prefix_spec.stages.back().heads.output->params.front().value = std::int64_t{32};
  llmp::spec::Canonicalize(prefix_spec);
  const auto prefix_path =
      publish_v1(store_dir, "unregistered-drafter", prefix_spec, "safetensors", 32);
  for (const auto requested : {std::optional<std::int64_t>{}, std::optional<std::int64_t>{0},
                               std::optional<std::int64_t>{16}, std::optional<std::int64_t>{64}}) {
    auto entry = composed_entry;
    entry.drafter = prefix_path.filename().string();
    if (requested) entry.overrides["draft_vocab"] = *requested;
    llmp::runtime::OpenedModelArtifacts retained;
    const auto facts = llmp::runtime::ReadArtifactFacts(entry, store_dir, &retained);
    ASSERT_TRUE(facts) << facts.error();
    ASSERT_TRUE(facts->composed_spec) << facts->composed_spec_error;
    ASSERT_TRUE(retained.composition);
    const auto expected = requested && *requested == 16 ? 16U : 32U;
    EXPECT_EQ(facts->drafter_head_rows, 32U);
    EXPECT_EQ(retained.composition->vocabulary(), expected);
    const auto settings = ResolveSettings(entry, *facts, nullptr, false);
    ASSERT_TRUE(settings) << settings.error();
    EXPECT_EQ(settings->draft_vocab.value, expected);
  }
  const auto alias_target = llmp::artifact::Artifact::Open(
      publish_v1(store_dir, "qwen3_8_flash_next", ds_spec, "safetensors"));
  ASSERT_TRUE(alias_target) << alias_target.error().ToString();
  for (const bool wrong_width : {false, true}) {
    auto invalid = drafter_spec;
    if (wrong_width)
      invalid.stages.back().stream.width = 31;
    else
      invalid.stages.back().heads.output->params.front().value = std::int64_t{65};
    llmp::spec::Canonicalize(invalid);
    const auto drafter = llmp::artifact::Artifact::Open(
        publish_v1(store_dir, "qwen4exp-mtp", invalid, "safetensors"));
    ASSERT_TRUE(drafter) << drafter.error().ToString();
    const auto facts = llmp::runtime::ArtifactFactsOf(*alias_target, &*drafter);
    ASSERT_TRUE(facts) << facts.error();
    EXPECT_FALSE(facts->composed_spec);
    EXPECT_FALSE(facts->composed_spec_error.empty());
    EXPECT_THAT(facts->composed_spec_error,
                ::testing::Not(HasSubstr("does not target architecture")));
    ModelEntry bad_shape = Model("bad-shape", true);
    bad_shape.overrides["draft_vocab"] = std::int64_t{64};
    EXPECT_FALSE(ResolveSettings(bad_shape, *facts, nullptr, false));
  }

  // Multiple aliases retain independent parsed artifacts/specs while the
  // startup parser workspace is reused. Their loans must exist before
  // admission, survive factory handoff, and unwind on every failed alias.
  namespace rt = llmp::runtime;
  llmp::catalog::Catalog catalog;
  const auto domain = catalog.AddDomain("pending aliases");
  std::uint64_t cap = UINT64_MAX;
  const auto occupancy = [&] { return catalog.OccupancyOf(domain).Total().value(); };
  std::uint64_t funding_calls = 0;
  const rt::StartupArtifactFunder funder = [&](const llmp::artifact::Artifact& target,
                                               const llmp::artifact::Artifact* draft)
      -> std::expected<std::unique_ptr<rt::StartupArtifactPayment>, std::string> {
    ++funding_calls;
    auto bytes = rt::StartupModelBytes(target, draft);
    if (!bytes) return std::unexpected(bytes.error());
    return rt::StartupArtifactPayment::Admit(catalog, domain, *bytes, cap, 128ULL << 30U);
  };
  std::vector<std::shared_ptr<rt::OpenedModelArtifacts>> pending;
  for (unsigned alias = 0; alias < 2; ++alias) {
    auto owner = std::make_shared<rt::OpenedModelArtifacts>();
    const auto facts = rt::ReadArtifactFacts(composed_entry, store_dir, owner.get(), funder);
    ASSERT_TRUE(facts) << (facts ? "" : facts.error());
    ASSERT_TRUE(owner->payment);
    EXPECT_TRUE(owner->target_admission);
    EXPECT_TRUE(owner->drafter_admission);
    if (alias == 0) cap = 2 * owner->payment->bytes();
    pending.push_back(std::move(owner));
  }
  EXPECT_EQ(occupancy(), cap);
  rt::OpenedModelArtifacts blocked;
  const auto refused = rt::ReadArtifactFacts(composed_entry, store_dir, &blocked, funder);
  ASSERT_FALSE(refused);
  EXPECT_THAT(refused.error(), HasSubstr("retained startup artifacts"));
  EXPECT_FALSE(blocked.target_admission);  // funding refused before spec/source admission
  EXPECT_FALSE(blocked.payment);
  EXPECT_EQ(occupancy(), cap);

  auto factory = std::move(pending.front());
  pending.clear();
  EXPECT_EQ(occupancy(), factory->payment->bytes());
  auto malformed = ds_spec;
  malformed.stages.front().context = 0;
  auto bad_entry = composed_entry;
  bad_entry.artifact = publish_v1(store_dir, "deepseek4", malformed).filename().string();
  const auto before_failure = occupancy();
  {
    rt::OpenedModelArtifacts failed;
    const auto bad = rt::ReadArtifactFacts(bad_entry, store_dir, &failed, funder);
    ASSERT_FALSE(bad);
    ASSERT_TRUE(failed.payment);  // failure happened after funding, during admission
    EXPECT_GT(occupancy(), before_failure);
  }
  EXPECT_EQ(occupancy(), before_failure);
  EXPECT_EQ(funding_calls, 4U);
  // An aliasing read-only artifact view carries the whole paid preparation.
  // It remains usable after factory/alias scopes disappear, then releases the
  // last payment after its artifacts, admissions and composition are destroyed.
  auto view = std::shared_ptr<const llmp::artifact::Artifact>(factory, &*factory->target);
  std::weak_ptr<const rt::OpenedModelArtifacts> owner_lifetime = factory;
  const auto* composition_address = factory->composition.get();
  auto second_factory = factory;
  factory.reset();
  EXPECT_EQ(second_factory->composition.get(), composition_address);
  second_factory.reset();
  EXPECT_FALSE(owner_lifetime.expired());
  EXPECT_EQ(view->id(), *composed_entry.artifact);
  EXPECT_EQ(occupancy(), before_failure);
  view.reset();
  EXPECT_TRUE(owner_lifetime.expired());
  EXPECT_EQ(occupancy(), 0U);
}

// Later layers win: an override over a calibration over a derived default
// over the fallback.
TEST(ModelSettings, LayersResolveInOrder) {
  Calibration calibration;
  calibration.prefill_chunk = 2048;
  calibration.max_slots = 6;
  calibration.decode_floor_tok_s = 12;
  calibration.wave_costs.fill(1.5);
  calibration.basis = "measured";
  ModelEntry entry = Model("ds", true);
  entry.overrides["max_slots"] = std::int64_t{2};
  entry.overrides["temperature"] = 0.0;
  ArtifactFacts facts = Facts("deepseek4", "dflash");
  facts.temperature = 0.7;
  const ModelSettings s = Resolved(entry, facts, &calibration);
  EXPECT_EQ(s.prefill_chunk.value, 2048U);
  EXPECT_EQ(s.prefill_chunk.source, SettingSource::kCalibrated);
  EXPECT_EQ(s.prefill_chunk.basis, "measured");
  EXPECT_EQ(s.max_slots.value, 2U);
  EXPECT_EQ(s.max_slots.source, SettingSource::kOverride);
  EXPECT_EQ(s.max_slots.basis, "models.ds.max_slots");
  EXPECT_EQ(s.decode_floor_tok_s.value, 12U);
  EXPECT_EQ(s.decode_floor_tok_s.source, SettingSource::kCalibrated);
  EXPECT_EQ(s.prefill_floor_tok_s.source, SettingSource::kFallback);
  EXPECT_EQ(s.temperature.value, 0.0);
  EXPECT_EQ(s.temperature.source, SettingSource::kOverride);
  EXPECT_EQ(s.wave_costs.value, std::vector<double>(7, 1.5));
  EXPECT_EQ(s.wave_costs.source, SettingSource::kCalibrated);

  // The widths measured; the rest keep their fallback.
  calibration.wave_costs = {};
  calibration.wave_costs[1] = 1.25;
  const ModelSettings some = Resolved(entry, facts, &calibration);
  EXPECT_EQ(some.wave_costs.source, SettingSource::kCalibrated);
  EXPECT_THAT(some.wave_costs.basis, HasSubstr("widths 3 measured"));
  EXPECT_EQ(some.wave_costs.value[1], 1.25);
  EXPECT_EQ(some.wave_costs.value[0], llmp::runtime::RecipeOf("deepseek4")->default_wave_costs[0]);
  calibration.wave_costs = {};
  EXPECT_EQ(Resolved(entry, facts, &calibration).wave_costs.source, SettingSource::kFallback);

  // A partial override over a calibration: the widths it names, then the
  // widths calibrated, then the fallback.
  {
    Calibration measured;
    measured.wave_costs[0] = 1.5;
    measured.wave_costs[2] = 2.5;
    measured.basis = "measured";
    ModelEntry partial = Model("ds", true);
    partial.overrides["wave_costs"] = std::vector<double>{3.0};
    const ModelSettings p = Resolved(partial, facts, &measured);
    EXPECT_EQ(p.wave_costs.source, SettingSource::kOverride);
    EXPECT_EQ(p.wave_costs_override_count, 1U);
    ASSERT_EQ(p.wave_costs.value.size(), 7U);
    EXPECT_EQ(p.wave_costs.value[0], 3.0);
    EXPECT_EQ(p.wave_costs.value[1], llmp::runtime::RecipeOf("deepseek4")->default_wave_costs[1]);
    EXPECT_EQ(p.wave_costs.value[2], 2.5);
    EXPECT_THAT(p.wave_costs.basis, HasSubstr("widths 2, 4 measured"));
  }

  // Widths an override names replace their fallbacks; the rest stay.
  entry.overrides["wave_costs"] = std::vector<double>{0.0, 3.0};
  const ModelSettings w = Resolved(entry, facts);
  EXPECT_EQ(w.wave_costs.source, SettingSource::kOverride);
  EXPECT_EQ(w.wave_costs_override_count, 2U);
  ASSERT_EQ(w.wave_costs.value.size(), 7U);
  EXPECT_EQ(w.wave_costs.value[0], 0.0);
  EXPECT_EQ(w.wave_costs.value[1], 3.0);
  EXPECT_EQ(w.wave_costs.value[2], llmp::runtime::RecipeOf("deepseek4")->default_wave_costs[2]);
}

// The context within the checkpoint's ceiling; DeepSeek's chunk past
// 262,144 tokens; its slots with DSpark at most eight; --plain.
TEST(ModelSettings, BoundsFromTheArchitecture) {
  ModelEntry entry = Model("q");
  entry.overrides["context"] = std::int64_t{262145};
  auto refused = ResolveSettings(entry, Facts("qwen4exp"), nullptr, false);
  ASSERT_FALSE(refused.has_value());
  EXPECT_THAT(refused.error(), HasSubstr("qwen4exp's supported range 512 to 262144"));
  ArtifactFacts small = Facts("deepseek4");
  small.trained_context = 65536;
  small.trained_context_from = "deepseek4.context_length";
  entry.overrides["context"] = std::int64_t{65537};
  refused = ResolveSettings(entry, small, nullptr, false);
  ASSERT_FALSE(refused.has_value());
  EXPECT_THAT(refused.error(), HasSubstr("512 to 65536 (deepseek4.context_length 65536)"));
  EXPECT_FALSE(ResolveSettings(Model("x"), Facts("llama"), nullptr, false).has_value());

  entry.overrides["context"] = std::int64_t{1048576};
  EXPECT_EQ(Resolved(entry, Facts("deepseek4")).prefill_chunk.value,
            llmp::runtime::RecipeOf("deepseek4")->deep_prefill_rows);
  entry.overrides["context"] = std::int64_t{262144};
  EXPECT_EQ(Resolved(entry, Facts("deepseek4")).prefill_chunk.value,
            llmp::runtime::RecipeOf("deepseek4")->prefill_rows);

  ModelEntry wide = Model("ds", true);
  wide.overrides["max_slots"] = std::int64_t{16};
  const ModelSettings capped = Resolved(wide, Facts("deepseek4", "dflash"));
  EXPECT_EQ(capped.max_slots.value, 16U);
  EXPECT_EQ(capped.max_slots.source, SettingSource::kOverride);
  EXPECT_THAT(capped.max_slots.basis, HasSubstr("max_slots"));
  const ModelSettings plain = Resolved(wide, Facts("deepseek4", "dflash"), nullptr, true);
  EXPECT_FALSE(plain.speculation.value);
  EXPECT_EQ(plain.speculation.basis, "--plain");
  EXPECT_EQ(plain.max_slots.value, 16U);
  ModelEntry off = Model("ds", true);
  off.overrides["speculation"] = false;
  EXPECT_EQ(Resolved(off, Facts("deepseek4", "dflash")).max_slots.value, 4U);
  EXPECT_EQ(Resolved(off, Facts("deepseek4", "dflash")).speculation.source,
            SettingSource::kOverride);
}

// An override the artifact cannot honour is refused; one the architecture
// does not use is ignored, and said.
TEST(ModelSettings, RefusesAndIgnoresOverrides) {
  ArtifactFacts facts = Facts("deepseek4", "dflash");
  facts.drafter_block = 5;
  ModelEntry entry = Model("ds", true);
  entry.overrides["draft_rows"] = std::int64_t{6};
  auto refused = ResolveSettings(entry, facts, nullptr, false);
  ASSERT_FALSE(refused.has_value());
  EXPECT_THAT(refused.error(), HasSubstr("stored block proposal bound"));
  entry.overrides["draft_rows"] = std::int64_t{5};
  EXPECT_EQ(Resolved(entry, facts).draft_rows.value, 5U);
  // A top_p the sampler's float would make 0 (the schema's range is open
  // at 0 as a double's).
  entry.overrides["top_p"] = 1e-50;
  EXPECT_FALSE(ResolveSettings(entry, facts, nullptr, false).has_value());
  entry.overrides.erase("top_p");

  ModelEntry q = Model("q", true);
  q.overrides["draft_rows"] = std::int64_t{1};
  EXPECT_TRUE(ResolveSettings(q, Facts("qwen4exp", "qwen4exp-mtp"), nullptr, false).has_value());
  q.overrides["draft_rows"] = std::int64_t{2};
  q.overrides["wave_form"] = std::string("plain");
  q.overrides["wave_costs"] = std::vector<double>{1.0};
  const ModelSettings s = Resolved(q, Facts("qwen4exp", "qwen4exp-mtp"));
  EXPECT_EQ(s.draft_rows.value, 2U);
  // Adaptive wave selection is shared by actual speculative compositions;
  // Qwen no longer loses its explicit controls to a family-only allowlist.
  EXPECT_TRUE(s.ignored.empty());
  EXPECT_EQ(s.wave_form.value, llmp::config::WaveForm::kPlain);
  EXPECT_EQ(s.wave_form.source, SettingSource::kOverride);
  EXPECT_EQ(s.wave_costs.source, SettingSource::kOverride);
  EXPECT_EQ(s.wave_costs_override_count, 1U);
  ASSERT_FALSE(s.wave_costs.value.empty());
  EXPECT_EQ(s.wave_costs.value.front(), 1.0);

  ModelEntry lone = Model("lone");
  lone.overrides["speculation"] = true;
  const ModelSettings no_drafter = Resolved(lone, Facts("qwen4exp"));
  EXPECT_FALSE(no_drafter.speculation.value);
  EXPECT_THAT(no_drafter.ignored, Contains(HasSubstr("speculation")));
}

// The listing shows what is in effect: a wave's drafts at most a step's.
TEST(ModelSettings, ListsEffectiveValues) {
  ModelEntry q = Model("q", true);
  q.overrides["draft_rows"] = std::int64_t{2};
  q.overrides["shared_wave_depth"] = std::int64_t{3};
  const ModelSettings s = Resolved(q, Facts("qwen4exp", "qwen4exp-mtp"));
  EXPECT_EQ(s.shared_wave_depth.value, 2U);
  EXPECT_EQ(s.shared_wave_depth.source, SettingSource::kOverride);
  EXPECT_THAT(s.shared_wave_depth.basis, HasSubstr("at most draft_rows"));
}

// The reasoning markers: an override that is a token, else the
// vocabulary's <think> pair, else none.
TEST(ModelSettings, ResolvesTheReasoningMarkers) {
  const std::map<std::string, std::int32_t, std::less<>> vocabulary = {
      {"<think>", 10}, {"</think>", 11}, {"<|thought|>", 12}, {"<|/thought|>", 13}};
  const auto find = [&](std::string_view text) -> std::optional<std::int32_t> {
    const auto it = vocabulary.find(text);
    return it == vocabulary.end() ? std::nullopt : std::optional(it->second);
  };
  ModelEntry entry = Model("q");
  ModelSettings s = Resolved(entry, Facts("qwen4exp"));
  EXPECT_EQ(s.reasoning_start.source, SettingSource::kFallback);
  ASSERT_TRUE(ResolveReasoning(s, entry, find).has_value());
  EXPECT_EQ(s.reasoning_start.value, "<think>");
  EXPECT_EQ(s.reasoning_end.value, "</think>");
  EXPECT_EQ(s.reasoning_start.source, SettingSource::kDerived);

  const auto none = [](std::string_view) -> std::optional<std::int32_t> { return std::nullopt; };
  ASSERT_TRUE(ResolveReasoning(s, entry, none).has_value());
  EXPECT_EQ(s.reasoning_start.value, "");
  EXPECT_EQ(s.reasoning_start.source, SettingSource::kFallback);

  entry.overrides["reasoning_start"] = std::string("<|thought|>");
  entry.overrides["reasoning_end"] = std::string("<|/thought|>");
  ASSERT_TRUE(ResolveReasoning(s, entry, find).has_value());
  EXPECT_EQ(s.reasoning_start.value, "<|thought|>");
  EXPECT_EQ(s.reasoning_end.source, SettingSource::kOverride);

  entry.overrides["reasoning_start"] = std::string("");  // none
  ASSERT_TRUE(ResolveReasoning(s, entry, find).has_value());
  EXPECT_EQ(s.reasoning_start.value, "");
  EXPECT_EQ(s.reasoning_start.source, SettingSource::kOverride);

  const auto original_start = s.reasoning_start;
  const auto original_end = s.reasoning_end;
  entry.overrides["reasoning_start"] = std::string("<think>");
  entry.overrides["reasoning_end"] = std::string("<nope>");
  auto refused = ResolveReasoning(s, entry, find);
  ASSERT_FALSE(refused.has_value());
  EXPECT_THAT(refused.error(), HasSubstr("not a token"));
  EXPECT_EQ(s.reasoning_start.value, original_start.value);
  EXPECT_EQ(s.reasoning_start.source, original_start.source);
  EXPECT_EQ(s.reasoning_start.basis, original_start.basis);
  EXPECT_EQ(s.reasoning_end.value, original_end.value);
  EXPECT_EQ(s.reasoning_end.source, original_end.source);
  EXPECT_EQ(s.reasoning_end.basis, original_end.basis);
}

// The listing is the schema's: every setting a model uses, once, in
// ModelKeys' order; every setting key is used by some kind of model.
TEST(ModelSettings, ListsEverySettingOfTheSchema) {
  std::set<std::string_view> listed;
  const auto keys_of = [&](const ModelSettings& s, std::string_view architecture,
                           bool composition) {
    std::vector<std::string_view> expected;
    for (const auto& spec : llmp::config::ModelKeys()) {
      const auto kind =
          composition ? llmp::config::kCompositionModels : llmp::config::kArtifactModels;
      if (spec.setting && (spec.kinds & kind) != 0 && llmp::runtime::KeyAppliesTo(spec, s)) {
        expected.push_back(spec.key);
      }
    }
    std::vector<std::string_view> got;
    for (const auto& line : s.Lines()) {
      got.push_back(line.key);
      listed.insert(line.key);
      EXPECT_FALSE(line.value.empty()) << line.key;
      EXPECT_FALSE(line.basis.empty()) << line.key;
    }
    EXPECT_EQ(got, expected) << architecture;
  };
  keys_of(Resolved(Model("ds", true), Facts("deepseek4", "dflash")), "deepseek4", false);
  keys_of(Resolved(Model("q", true), Facts("qwen4exp", "qwen4exp-mtp")), "qwen4exp", false);
  ModelEntry image;
  image.name = "image";
  image.composition = std::string(64, 'c');
  image.overrides["image_steps"] = std::int64_t{20};
  ArtifactFacts pipeline;
  pipeline.composition = true;
  const ModelSettings img = Resolved(image, pipeline);
  EXPECT_EQ(img.image_steps.value, 20U);
  EXPECT_EQ(img.image_steps.source, SettingSource::kOverride);
  EXPECT_EQ(img.image_size.value, 1024U);
  keys_of(img, "", true);
  for (const auto& spec : llmp::config::ModelKeys()) {
    if (spec.setting) {
      EXPECT_TRUE(listed.contains(spec.key)) << spec.key << " is in no model's listing";
    }
  }
  const ModelSettings s = Resolved(Model("ds"), Facts("deepseek4"));
  EXPECT_THAT(s.Summary(), HasSubstr("context=262144 (fallback)"));
  EXPECT_THAT(s.Summary(), HasSubstr("wave_form=\"auto\" (fallback)"));
  EXPECT_THAT(s.Summary(), HasSubstr("speculation=false (derived)"));
}

TEST(ModelSettings, CapabilityPredicatesAndKeyApplicability) {
  using llmp::runtime::KeyAppliesTo;
  using llmp::runtime::SupportsMtpDrafting;
  using llmp::runtime::SupportsSpeculativeDrafting;
  using llmp::runtime::SupportsWaveCosts;
  using llmp::runtime::SupportsWaveForm;

  ArtifactFacts ds_facts = Facts("deepseek4");
  EXPECT_TRUE(SupportsSpeculativeDrafting(ds_facts));
  EXPECT_TRUE(SupportsWaveForm(ds_facts));
  EXPECT_TRUE(SupportsWaveCosts(ds_facts));
  EXPECT_FALSE(SupportsMtpDrafting(ds_facts));
  EXPECT_TRUE(KeyAppliesTo("draft_rows", ds_facts));
  EXPECT_TRUE(KeyAppliesTo("wave_form", ds_facts));
  EXPECT_TRUE(KeyAppliesTo("wave_costs", ds_facts));
  EXPECT_FALSE(KeyAppliesTo("draft_vocab", ds_facts));
  EXPECT_FALSE(KeyAppliesTo("image_size", ds_facts));

  ArtifactFacts qwen_facts = Facts("qwen4exp", "qwen4exp-mtp");
  EXPECT_TRUE(SupportsSpeculativeDrafting(qwen_facts));
  EXPECT_TRUE(SupportsWaveForm(qwen_facts));
  EXPECT_TRUE(SupportsWaveCosts(qwen_facts));
  EXPECT_TRUE(SupportsMtpDrafting(qwen_facts));
  EXPECT_TRUE(KeyAppliesTo("draft_rows", qwen_facts));
  EXPECT_TRUE(KeyAppliesTo("wave_form", qwen_facts));
  EXPECT_TRUE(KeyAppliesTo("draft_vocab", qwen_facts));
  EXPECT_TRUE(KeyAppliesTo("depth_cost_ratio", qwen_facts));

  ArtifactFacts gemma_facts = Facts("gemma4");
  EXPECT_FALSE(SupportsSpeculativeDrafting(gemma_facts));
  EXPECT_FALSE(SupportsWaveForm(gemma_facts));
  EXPECT_FALSE(SupportsWaveCosts(gemma_facts));
  EXPECT_FALSE(SupportsMtpDrafting(gemma_facts));
  EXPECT_FALSE(KeyAppliesTo("draft_rows", gemma_facts));
  EXPECT_FALSE(KeyAppliesTo("wave_form", gemma_facts));
  EXPECT_FALSE(KeyAppliesTo("draft_vocab", gemma_facts));
  EXPECT_TRUE(KeyAppliesTo("context", gemma_facts));

  ArtifactFacts gemma_drafted = Facts("gemma4", "gemma4-assistant");
  EXPECT_TRUE(SupportsSpeculativeDrafting(gemma_drafted));
  EXPECT_TRUE(KeyAppliesTo("draft_rows", gemma_drafted));

  ArtifactFacts image_facts;
  image_facts.composition = true;
  EXPECT_TRUE(KeyAppliesTo("image_size", image_facts));
  EXPECT_TRUE(KeyAppliesTo("image_steps", image_facts));
  EXPECT_FALSE(KeyAppliesTo("context", image_facts));
  EXPECT_FALSE(KeyAppliesTo("draft_rows", image_facts));
}

TEST(ModelSettings, DraftVocabularyUsesTheActualHeadRows) {
  auto facts = Facts("qwen4exp", "qwen4exp-mtp");
  facts.vocabulary = 100000;
  facts.drafter_head_rows = 100000;
  auto entry = Model("other-size", true);
  entry.overrides["draft_vocab"] = std::int64_t{100001};
  const auto capped = ResolveSettings(entry, facts, nullptr, false);
  ASSERT_TRUE(capped) << capped.error();
  EXPECT_EQ(capped->draft_vocab.value, 100000U);
  entry.overrides["draft_vocab"] = std::int64_t{100000};
  EXPECT_TRUE(ResolveSettings(entry, facts, nullptr, false));
}

}  // namespace
