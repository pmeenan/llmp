// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// A model's settings in three layers (D-103, runtime/model_settings.h):
// each from the artifact's facts, a calibration or an override, else its
// fallback, with its source; refused overrides; the reasoning markers; and
// the listing against the configuration's table-driven schema.

#include "runtime/model_settings.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "config/node_config.h"
#include "model/gemma4.h"

namespace {

using ::jitllm::config::ModelEntry;
using ::jitllm::runtime::ArtifactFacts;
using ::jitllm::runtime::Calibration;
using ::jitllm::runtime::ModelSettings;
using ::jitllm::runtime::ResolveReasoning;
using ::jitllm::runtime::ResolveSettings;
using ::jitllm::runtime::SettingSource;
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
  facts.drafter_architecture = std::move(drafter);
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
TEST(ModelSettings, FallbacksWhenTheArtifactSaysNothing) {
  const ModelSettings s = Resolved(Model("ds"), Facts("deepseek4"));
  EXPECT_EQ(s.context.value, 262144U);
  EXPECT_EQ(s.context.source, SettingSource::kFallback);
  EXPECT_FALSE(s.speculation.value);
  EXPECT_EQ(s.speculation.source, SettingSource::kDerived);  // no drafter
  EXPECT_EQ(s.prefill_chunk.value, jitllm::runtime::kDsv4PrefillRows);
  EXPECT_EQ(s.prefill_chunk.source, SettingSource::kFallback);
  EXPECT_EQ(s.max_slots.value, 4U);
  EXPECT_EQ(s.prefill_floor_tok_s.value, 100U);
  EXPECT_EQ(s.decode_floor_tok_s.value, 5U);
  EXPECT_EQ(s.recompute_ms_per_token.value, 1.4);
  EXPECT_EQ(s.temperature.value, 1.0);
  EXPECT_EQ(s.temperature.source, SettingSource::kFallback);
  EXPECT_EQ(s.top_k.value, 0U);
  EXPECT_EQ(s.draft_rows.value, 3U);
  EXPECT_EQ(s.wave_form.value, jitllm::config::WaveForm::kAuto);
  EXPECT_EQ(s.wave_costs.value, (std::vector<double>(jitllm::runtime::kDsv4WaveCosts.begin(),
                                                     jitllm::runtime::kDsv4WaveCosts.end())));
  EXPECT_EQ(s.wave_costs_override_count, 0U);
  EXPECT_TRUE(s.prefill_outa_hca.value);
  EXPECT_TRUE(s.prefill_outa_hca_partial.value);
  EXPECT_TRUE(s.ignored.empty());

  const ModelSettings q = Resolved(Model("q"), Facts("qwen4exp"));
  EXPECT_EQ(q.context.value, 262144U);
  EXPECT_EQ(q.prefill_chunk.value, jitllm::runtime::kQwen38PrefillRows);
  EXPECT_EQ(q.draft_vocab.value, 65536U);
  EXPECT_EQ(q.depth_cost_ratio.value, 1.16);
  EXPECT_EQ(q.shared_wave_depth.value, 2U);
  EXPECT_EQ(q.draft_wave_max.value, 2U);
  EXPECT_EQ(q.wave_read_align.value, 2048U);
  EXPECT_TRUE(q.wave_lanes.value);
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
  EXPECT_FALSE(ResolveSettings(oversized, facts, nullptr, false).has_value());
  const ModelSettings prefix = Resolved(Model("q", true), Facts("qwen4exp", "qwen4exp-mtp"));
  EXPECT_EQ(prefix.draft_vocab.value, 65536U);
  EXPECT_EQ(prefix.draft_vocab.source, SettingSource::kFallback);
  const ModelSettings plain = Resolved(Model("q", true), facts, nullptr, true);
  EXPECT_FALSE(plain.speculation.value);
  EXPECT_EQ(plain.draft_vocab.value, 65536U);
}

TEST(ModelSettings, OwnerCanDisableConcurrentWaveLanes) {
  ModelEntry entry = Model("q");
  entry.overrides["wave_lanes"] = false;
  const ModelSettings q = Resolved(entry, Facts("qwen4exp"));
  EXPECT_FALSE(q.wave_lanes.value);
  EXPECT_EQ(q.wave_lanes.source, SettingSource::kOverride);
  EXPECT_THAT(q.Summary(), HasSubstr("wave_lanes=false (override)"));
}

TEST(ModelSettings, GemmaDefaultsAreBoundedAndExplicitlyUncalibrated) {
  auto entry = Model("gemma");
  auto facts = Facts("gemma4");
  facts.trained_context = 262144;
  facts.trained_context_from = "gemma4.context_length";
  const auto s = Resolved(entry, facts);
  EXPECT_EQ(s.context.value, 262144U);
  EXPECT_EQ(s.max_slots.value, 1U);
  EXPECT_EQ(s.prefill_chunk.value, 128U);
  EXPECT_THAT(s.max_slots.basis, HasSubstr("uncalibrated"));
  EXPECT_THAT(s.prefill_chunk.basis, HasSubstr("uncalibrated"));
  EXPECT_THAT(s.recompute_ms_per_token.basis, HasSubstr("uncalibrated"));
  EXPECT_FALSE(s.speculation.value);
  for (const auto owners : {1, 2, 4, 8, 12}) {
    entry.overrides["max_slots"] = std::int64_t{owners};
    const auto chosen = Resolved(entry, facts);
    EXPECT_EQ(chosen.max_slots.value, static_cast<std::uint32_t>(owners));
  }
  entry.overrides["max_slots"] = std::int64_t{13};
  EXPECT_FALSE(ResolveSettings(entry, facts, nullptr, false));
  entry.overrides["max_slots"] = std::int64_t{12};
  entry.overrides["prefill_chunk"] = std::int64_t{4};
  EXPECT_FALSE(ResolveSettings(entry, facts, nullptr, false));
  entry.overrides["prefill_chunk"] = std::int64_t{4096};
  EXPECT_EQ(Resolved(entry, facts).prefill_chunk.value, 128U);
  entry.overrides["context"] = std::int64_t{262145};
  EXPECT_FALSE(ResolveSettings(entry, facts, nullptr, false));
  EXPECT_FALSE(ResolveSettings(Model("gemma", true), facts, nullptr, false));
  entry = Model("gemma");
  entry.overrides["speculation"] = true;
  EXPECT_FALSE(ResolveSettings(entry, facts, nullptr, false));
}

TEST(ModelSettings, Gemma31ProductionDefaultIsBoundedAndProfileSpecific) {
  auto entry = Model("gemma");
  entry.overrides["context"] = std::int64_t{8192};
  entry.overrides["prefill_chunk"] = std::int64_t{1024};
  entry.overrides["max_slots"] = std::int64_t{4};
  auto facts = Facts("gemma4");
  facts.gemma_profile = &jitllm::model::Gemma4_31B();
  auto ordinary = ResolveSettings(entry, facts, nullptr, true, false);
  ASSERT_TRUE(ordinary);
  EXPECT_FALSE(ordinary->gemma31_production);
  EXPECT_EQ(ordinary->prefill_chunk.value, 128U);
  auto candidate = ResolveSettings(entry, facts, nullptr, true);
  ASSERT_TRUE(candidate);
  EXPECT_TRUE(candidate->gemma31_production);
  EXPECT_EQ(candidate->prefill_chunk.value, 256U);
  EXPECT_EQ(candidate->max_slots.value, 4U);
  EXPECT_FALSE(candidate->speculation.value);
  entry.overrides["context"] = std::int64_t{8193};
  auto larger_context = ResolveSettings(entry, facts, nullptr, true);
  ASSERT_TRUE(larger_context);
  EXPECT_FALSE(larger_context->gemma31_production);
  EXPECT_EQ(larger_context->prefill_chunk.value, 128U);
  entry.overrides["context"] = std::int64_t{8192};
  entry.overrides["max_slots"] = std::int64_t{5};
  auto larger_cohort = ResolveSettings(entry, facts, nullptr, true);
  ASSERT_TRUE(larger_cohort);
  EXPECT_FALSE(larger_cohort->gemma31_production);
  EXPECT_EQ(larger_cohort->prefill_chunk.value, 128U);
  entry.overrides["max_slots"] = std::int64_t{4};
  facts.gemma_profile = &jitllm::model::Gemma4_26BA4B();
  auto profile26 = ResolveSettings(entry, facts, nullptr, true, true);
  ASSERT_TRUE(profile26);
  EXPECT_FALSE(profile26->gemma31_production);
  EXPECT_TRUE(profile26->gemma26_production);
  EXPECT_EQ(profile26->prefill_chunk.value, 1024U);
  facts.gemma_profile = nullptr;
  auto unknown = ResolveSettings(entry, facts, nullptr, true, true);
  ASSERT_TRUE(unknown);
  EXPECT_FALSE(unknown->gemma31_production);
  EXPECT_EQ(unknown->prefill_chunk.value, 128U);
}

TEST(ModelSettings, Gemma26ProductionUses1024WithinBoundsAndPreservesSmallerOverride) {
  auto entry = Model("gemma");
  auto facts = Facts("gemma4");
  facts.gemma_profile = &jitllm::model::Gemma4_26BA4B();
  auto ordinary = ResolveSettings(entry, facts, nullptr, true, false);
  ASSERT_TRUE(ordinary);
  EXPECT_FALSE(ordinary->gemma26_production);
  // Default context exceeds the bounded envelope: the prior recipe.
  auto unbounded = ResolveSettings(entry, facts, nullptr, true);
  ASSERT_TRUE(unbounded);
  EXPECT_FALSE(unbounded->gemma26_production);
  EXPECT_EQ(unbounded->prefill_chunk.value, 128U);
  entry.overrides["context"] = std::int64_t{8192};
  auto candidate = ResolveSettings(entry, facts, nullptr, true);
  ASSERT_TRUE(candidate);
  EXPECT_TRUE(candidate->gemma26_production);
  EXPECT_FALSE(candidate->gemma31_production);
  EXPECT_EQ(candidate->prefill_chunk.value, 1024U);
  entry.overrides["max_slots"] = std::int64_t{4};
  candidate = ResolveSettings(entry, facts, nullptr, true);
  ASSERT_TRUE(candidate);
  EXPECT_TRUE(candidate->gemma26_production);
  entry.overrides["max_slots"] = std::int64_t{5};
  auto larger_cohort = ResolveSettings(entry, facts, nullptr, true);
  ASSERT_TRUE(larger_cohort);
  EXPECT_FALSE(larger_cohort->gemma26_production);
  EXPECT_EQ(larger_cohort->prefill_chunk.value, 128U);
  entry.overrides["max_slots"] = std::int64_t{4};
  entry.overrides["context"] = std::int64_t{8193};
  auto larger_context = ResolveSettings(entry, facts, nullptr, true);
  ASSERT_TRUE(larger_context);
  EXPECT_FALSE(larger_context->gemma26_production);
  entry.overrides["context"] = std::int64_t{8192};
  entry.overrides["prefill_chunk"] = std::int64_t{2048};
  candidate = ResolveSettings(entry, facts, nullptr, true);
  ASSERT_TRUE(candidate);
  EXPECT_EQ(candidate->prefill_chunk.value, 1024U);
  entry.overrides["prefill_chunk"] = std::int64_t{64};
  candidate = ResolveSettings(entry, facts, nullptr, true);
  ASSERT_TRUE(candidate);
  EXPECT_TRUE(candidate->gemma26_production);
  EXPECT_EQ(candidate->prefill_chunk.value, 64U);
}

TEST(ModelSettings, Gemma31ProductionUses256WithinBoundsAndPreservesSmallerOverride) {
  auto entry = Model("gemma");
  auto facts = Facts("gemma4");
  facts.gemma_profile = &jitllm::model::Gemma4_31B();
  auto candidate = ResolveSettings(entry, facts, nullptr, true);
  ASSERT_TRUE(candidate);
  EXPECT_FALSE(candidate->gemma31_production);
  EXPECT_EQ(candidate->prefill_chunk.value, 128U);
  EXPECT_EQ(candidate->max_slots.value, 1U);
  entry.overrides["context"] = std::int64_t{8192};
  candidate = ResolveSettings(entry, facts, nullptr, true);
  ASSERT_TRUE(candidate);
  EXPECT_TRUE(candidate->gemma31_production);
  EXPECT_EQ(candidate->prefill_chunk.value, 256U);
  entry.overrides["prefill_chunk"] = std::int64_t{16};
  candidate = ResolveSettings(entry, facts, nullptr, true, true);
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

// The facts kept metadata gives: DeepSeek V4 Flash's GGUF keys as its
// artifact keeps them, a Qwen config.json's text_config, rope scaling,
// generation_config.json's sampling defaults (ahead of a GGUF's), and
// sampling defaults no request could ask for ignored.
TEST(ModelSettings, ReadsFactsFromMetadata) {
  using jitllm::artifact::GgufValue;
  const auto integer = [](std::int64_t v) {
    GgufValue g;
    g.kind = GgufValue::Kind::kInteger;
    g.integer = v;
    return g;
  };
  const auto real = [](double v) {
    GgufValue g;
    g.kind = GgufValue::Kind::kFloat;
    g.real = v;
    return g;
  };
  ArtifactFacts ds = Facts("deepseek4");
  EXPECT_EQ(jitllm::runtime::GgufFactKeys("deepseek4").front(), "deepseek4.context_length");
  jitllm::artifact::GgufMetadata gguf;
  gguf["deepseek4.context_length"] = integer(1048576);
  gguf["deepseek4.rope.scaling.factor"] = real(16.0);
  gguf["deepseek4.rope.scaling.original_context_length"] = integer(65536);
  gguf["general.sampling.temp"] = real(1.0);
  gguf["general.sampling.top_p"] = real(1.0);
  gguf["general.sampling.top_k"] = integer(-1);  // not a request's: ignored
  jitllm::runtime::FactsFromGguf(gguf, ds);
  EXPECT_EQ(ds.trained_context, 1048576U);
  EXPECT_EQ(ds.trained_context_from, "deepseek4.context_length");
  EXPECT_EQ(ds.temperature, 1.0);
  EXPECT_EQ(ds.top_p, 1.0);
  EXPECT_FALSE(ds.top_k.has_value());
  EXPECT_EQ(ds.sampling_from, "general.sampling");

  // A GGUF that states the original context: scaled.
  ArtifactFacts scaled = Facts("llama");
  jitllm::artifact::GgufMetadata original;
  original["llama.context_length"] = integer(32768);
  original["llama.rope.scaling.factor"] = real(4.0);
  original["llama.rope.scaling.original_context_length"] = integer(32768);
  jitllm::runtime::FactsFromGguf(original, scaled);
  EXPECT_EQ(scaled.trained_context, 131072U);
  EXPECT_THAT(scaled.trained_context_from, HasSubstr("rope.scaling.factor"));

  ArtifactFacts qwen = Facts("qwen4exp");
  ASSERT_TRUE(
      jitllm::runtime::FactsFromGenerationConfig(
          R"({"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":2.0,"do_sample":true})", qwen)
          .has_value());
  ASSERT_TRUE(jitllm::runtime::FactsFromConfigJson(
                  R"({"architectures":["X"],"text_config":{"max_position_embeddings":262144,)"
                  R"("rope_scaling":{"factor":2.0,"original_max_position_embeddings":32768}}})",
                  qwen)
                  .has_value());
  EXPECT_EQ(qwen.trained_context, 262144U);  // longer than 32,768 times 2
  EXPECT_EQ(qwen.trained_context_from, "config.json text_config.max_position_embeddings");
  EXPECT_EQ(qwen.temperature, 0.6);
  EXPECT_EQ(qwen.top_k, 20);
  EXPECT_FALSE(qwen.min_p.has_value());  // 2.0: not a request's
  // A GGUF's sampling after it adds nothing it already has.
  jitllm::runtime::FactsFromGguf(gguf, qwen);
  EXPECT_EQ(qwen.temperature, 0.6);
  EXPECT_EQ(qwen.sampling_from, "generation_config.json");

  ArtifactFacts top = Facts("qwen2");
  ASSERT_TRUE(jitllm::runtime::FactsFromConfigJson(R"({"max_position_embeddings":32768})", top)
                  .has_value());
  EXPECT_EQ(top.trained_context, 32768U);
  EXPECT_EQ(top.trained_context_from, "config.json max_position_embeddings");
  EXPECT_FALSE(jitllm::runtime::FactsFromConfigJson("{", top).has_value());
  EXPECT_FALSE(jitllm::runtime::FactsFromGenerationConfig("[1,", top).has_value());

  // A Qwen3.8 GGUF quantization keeps its checkpoint's sampling (the
  // UD-IQ3_XXS: top_k 20, top_p 0.95 as F32): requests that send neither
  // sample with them, and 0.95 stays 0.95, not the float's 0.949999988.
  jitllm::artifact::GgufMetadata qwen_gguf;
  GgufValue top_p_f32 = real(static_cast<double>(0.95F));
  top_p_f32.type = 6;  // F32
  qwen_gguf["general.sampling.top_p"] = top_p_f32;
  qwen_gguf["general.sampling.top_k"] = integer(20);
  qwen_gguf["qwen4exp.context_length"] = integer(262144);
  ArtifactFacts quant = Facts("qwen4exp");
  jitllm::runtime::FactsFromGguf(qwen_gguf, quant);
  EXPECT_EQ(quant.top_p, 0.95);
  EXPECT_EQ(quant.top_k, 20);
  const ModelSettings sampled = Resolved(Model("q"), quant);
  EXPECT_EQ(sampled.top_p.value, 0.95);
  EXPECT_EQ(sampled.top_p.source, SettingSource::kDerived);
  EXPECT_EQ(sampled.top_k.value, 20U);
  EXPECT_EQ(sampled.temperature.source, SettingSource::kFallback);
  EXPECT_THAT(sampled.Summary(), HasSubstr("top_p=0.95 (derived) top_k=20 (derived)"));

  // generation_config.json's do_sample false: greedy, whatever its
  // temperature.
  ArtifactFacts greedy = Facts("qwen4exp");
  ASSERT_TRUE(jitllm::runtime::FactsFromGenerationConfig(
                  R"({"temperature":0.6,"top_p":0.9,"do_sample":false})", greedy)
                  .has_value());
  EXPECT_EQ(greedy.temperature, 0.0);
  EXPECT_EQ(greedy.top_p, 0.9);
  EXPECT_THAT(greedy.sampling_from, HasSubstr("do_sample false"));
  ArtifactFacts sampling = Facts("qwen4exp");
  ASSERT_TRUE(jitllm::runtime::FactsFromGenerationConfig(R"({"temperature":0.6,"do_sample":true})",
                                                         sampling)
                  .has_value());
  EXPECT_EQ(sampling.temperature, 0.6);

  // Through resolution: the derived context below the default.
  ArtifactFacts small = Facts("qwen4exp");
  ASSERT_TRUE(jitllm::runtime::FactsFromConfigJson(
                  R"({"text_config":{"max_position_embeddings":40960}})", small)
                  .has_value());
  const ModelSettings s = Resolved(Model("q"), small);
  EXPECT_EQ(s.context.value, 40960U);
  EXPECT_EQ(s.context.source, SettingSource::kDerived);
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
  EXPECT_EQ(some.wave_costs.value[0], jitllm::runtime::kDsv4WaveCosts[0]);
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
    EXPECT_EQ(p.wave_costs.value[1], jitllm::runtime::kDsv4WaveCosts[1]);
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
  EXPECT_EQ(w.wave_costs.value[2], jitllm::runtime::kDsv4WaveCosts[2]);
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
            jitllm::runtime::kDsv4DeepPrefillRows);
  entry.overrides["context"] = std::int64_t{262144};
  EXPECT_EQ(Resolved(entry, Facts("deepseek4")).prefill_chunk.value,
            jitllm::runtime::kDsv4PrefillRows);

  ModelEntry wide = Model("ds", true);
  wide.overrides["max_slots"] = std::int64_t{16};
  const ModelSettings capped = Resolved(wide, Facts("deepseek4", "dflash"));
  EXPECT_EQ(capped.max_slots.value, 8U);
  EXPECT_EQ(capped.max_slots.source, SettingSource::kOverride);
  EXPECT_THAT(capped.max_slots.basis, HasSubstr("at most 8"));
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
  EXPECT_THAT(refused.error(), HasSubstr("the drafter's block of 5"));
  entry.overrides["draft_rows"] = std::int64_t{5};
  EXPECT_EQ(Resolved(entry, facts).draft_rows.value, 5U);
  // A top_p the sampler's float would make 0 (the schema's range is open
  // at 0 as a double's).
  entry.overrides["top_p"] = 1e-50;
  EXPECT_FALSE(ResolveSettings(entry, facts, nullptr, false).has_value());
  entry.overrides.erase("top_p");

  ModelEntry q = Model("q", true);
  q.overrides["draft_rows"] = std::int64_t{1};
  EXPECT_FALSE(ResolveSettings(q, Facts("qwen4exp", "qwen4exp-mtp"), nullptr, false).has_value());
  q.overrides["draft_rows"] = std::int64_t{2};
  q.overrides["wave_form"] = std::string("plain");
  q.overrides["prefill_outa_hca"] = false;
  const ModelSettings s = Resolved(q, Facts("qwen4exp", "qwen4exp-mtp"));
  EXPECT_EQ(s.draft_rows.value, 2U);
  EXPECT_THAT(s.ignored, Contains(HasSubstr("wave_form")));
  EXPECT_THAT(s.ignored, Contains(HasSubstr("prefill_outa_hca")));
  EXPECT_EQ(s.ignored.size(), 2U);

  ModelEntry lone = Model("lone");
  lone.overrides["speculation"] = true;
  const ModelSettings no_drafter = Resolved(lone, Facts("qwen4exp"));
  EXPECT_FALSE(no_drafter.speculation.value);
  EXPECT_THAT(no_drafter.ignored, Contains(HasSubstr("speculation")));
}

// The listing shows what is in effect: the partial output-A/HCA prefill
// only with the full-chunk form, a wave's drafts at most a step's.
TEST(ModelSettings, ListsEffectiveValues) {
  ModelEntry ds = Model("ds", true);
  ds.overrides["prefill_outa_hca"] = false;
  const ModelSettings d = Resolved(ds, Facts("deepseek4", "dflash"));
  EXPECT_FALSE(d.prefill_outa_hca.value);
  EXPECT_FALSE(d.prefill_outa_hca_partial.value);
  EXPECT_THAT(d.prefill_outa_hca_partial.basis, HasSubstr("prefill_outa_hca is off"));

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

  entry.overrides["reasoning_end"] = std::string("<nope>");
  auto refused = ResolveReasoning(s, entry, find);
  ASSERT_FALSE(refused.has_value());
  EXPECT_THAT(refused.error(), HasSubstr("not a token"));
}

// The listing is the schema's: every setting a model uses, once, in
// ModelKeys' order; every setting key is used by some kind of model.
TEST(ModelSettings, ListsEverySettingOfTheSchema) {
  std::set<std::string_view> listed;
  const auto keys_of = [&](const ModelSettings& s, std::string_view architecture,
                           bool composition) {
    std::vector<std::string_view> expected;
    for (const auto& spec : jitllm::config::ModelKeys()) {
      const auto kind =
          composition ? jitllm::config::kCompositionModels : jitllm::config::kArtifactModels;
      if (spec.setting && (spec.kinds & kind) != 0 &&
          (composition || jitllm::config::KeyAppliesTo(spec, architecture))) {
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
  for (const auto& spec : jitllm::config::ModelKeys()) {
    if (spec.setting) {
      EXPECT_TRUE(listed.contains(spec.key)) << spec.key << " is in no model's listing";
    }
  }
  const ModelSettings s = Resolved(Model("ds"), Facts("deepseek4"));
  EXPECT_THAT(s.Summary(), HasSubstr("context=262144 (fallback)"));
  EXPECT_THAT(s.Summary(), HasSubstr("wave_form=\"auto\" (fallback)"));
  EXPECT_THAT(s.Summary(), HasSubstr("speculation=false (derived)"));
}

}  // namespace
