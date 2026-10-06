// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// A model's calibration on this machine (D-103, runtime/calibration.h): the
// record's format, its key (stale records not used, foreign and corrupt
// ones refused), its atomic write and trusted read, and the measurements
// a model's first uses take.

#include "runtime/calibration.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

#include "runtime/prefill.h"

namespace {

namespace fs = std::filesystem;
using ::jitllm::runtime::Calibration;
using ::jitllm::runtime::CalibrationKey;
using ::jitllm::runtime::CalibrationSamples;
using ::jitllm::runtime::FormatCalibration;
using ::jitllm::runtime::ParseCalibration;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::StartsWith;

CalibrationKey Key() {
  return {.artifact = std::string(64, 'a'),
          .drafter = std::string(64, 'b'),
          .device = "NVIDIA GB10 sm_121, driver 580.95.05, CUDA 13.0",
          .build = "0.3.0-dev.1+g1234567",
          .settings =
              "speculation=true draft_rows=3 prefill_chunk=4096 max_slots=4 "
              "prefill_outa_hca=true/true wave_form=auto prefill_chunk_override=false "
              "max_slots_override=false context=0"};
}

Calibration Measured() {
  Calibration c;
  c.prefill_floor_tok_s = 333;
  c.decode_floor_tok_s = 7;
  c.recompute_ms_per_token = 1.25;
  c.wave_costs[0] = 2.18;
  c.wave_costs[2] = 2.99;
  c.depth_cost_ratio = 1.1;
  return c;
}

TEST(Calibration, RoundTripsItsRecord) {
  const std::string text = FormatCalibration(Measured(), Key());
  EXPECT_THAT(text, StartsWith(R"({"format":"jitllm-model-calibration-v1","artifact":")"));
  EXPECT_THAT(text, HasSubstr(R"("wave_costs":[2.18,null,2.99])"));
  const auto read = ParseCalibration(text, Key());
  ASSERT_TRUE(read.calibration.has_value()) << read.note;
  EXPECT_FALSE(read.refused);
  const Calibration c = read.calibration.value_or(Calibration{});
  EXPECT_EQ(c.prefill_floor_tok_s, 333U);
  EXPECT_EQ(c.decode_floor_tok_s, 7U);
  EXPECT_EQ(c.recompute_ms_per_token, 1.25);
  EXPECT_EQ(c.wave_costs[0], 2.18);
  EXPECT_FALSE(c.wave_costs[1].has_value());
  EXPECT_EQ(c.wave_costs[2], 2.99);
  EXPECT_EQ(c.depth_cost_ratio, 1.1);
  EXPECT_FALSE(c.prefill_chunk.has_value());
  EXPECT_THAT(c.basis, HasSubstr("measured on this machine"));

  CalibrationKey lone = Key();
  lone.drafter.clear();
  EXPECT_THAT(FormatCalibration(Calibration{}, lone), HasSubstr(R"("drafter":null)"));
  EXPECT_TRUE(ParseCalibration(FormatCalibration(Calibration{}, lone), lone).calibration);
}

// The settings a measurement depends on are part of the key: another
// configuration of them is another key.
TEST(Calibration, Gemma31CandidateHasDistinctCalibrationIdentity) {
  jitllm::runtime::ModelSettings settings;
  settings.architecture = "gemma4";
  const auto ordinary = jitllm::runtime::MeasuredWith(settings);
  settings.architecture.clear();
  EXPECT_EQ(ordinary, jitllm::runtime::MeasuredWith(settings));
  settings.architecture = "gemma4";
  settings.gemma31_production = true;
  const auto candidate = jitllm::runtime::MeasuredWith(settings);
  EXPECT_NE(ordinary, candidate);
  EXPECT_THAT(ordinary, Not(HasSubstr("gemma31_production=")));
  EXPECT_THAT(candidate, HasSubstr("gemma31_production=true"));
}

TEST(Calibration, KeysTheSettingsMeasuredWith) {
  jitllm::runtime::ModelSettings s;
  s.speculation.value = true;
  s.draft_rows.value = 3;
  s.prefill_chunk.value = 4096;
  s.max_slots.value = 4;
  s.prefill_outa_hca.value = true;
  s.prefill_outa_hca_partial.value = true;
  EXPECT_EQ(jitllm::runtime::MeasuredWith(s), Key().settings);
  s.speculation.value = false;  // --plain
  EXPECT_THAT(jitllm::runtime::MeasuredWith(s), StartsWith("speculation=false"));
  s.speculation.value = true;
  s.wave_form.value = jitllm::config::WaveForm::kPlain;
  EXPECT_THAT(jitllm::runtime::MeasuredWith(s), HasSubstr("wave_form=plain"));
  EXPECT_THAT(jitllm::runtime::BuildIdentity(), Not(IsEmpty()));
}

TEST(Calibration, ChangingQwenWaveLanesInvalidatesItsMeasurements) {
  jitllm::runtime::ModelSettings s;
  s.architecture = "qwen4exp";
  s.wave_lanes.value = true;
  CalibrationKey on = Key();
  on.settings = jitllm::runtime::MeasuredWith(s);
  s.wave_lanes.value = false;
  CalibrationKey off = on;
  off.settings = jitllm::runtime::MeasuredWith(s);
  EXPECT_NE(on.settings, off.settings);
  const auto read = ParseCalibration(FormatCalibration(Measured(), on), off);
  EXPECT_FALSE(read.calibration.has_value());
  EXPECT_FALSE(read.refused);
  EXPECT_THAT(read.note, StartsWith("stale"));
}

TEST(Calibration, ChangingQwenDraftOrWavePolicyInvalidatesItsMeasurements) {
  jitllm::runtime::ModelSettings measured;
  measured.architecture = "qwen4exp";
  measured.draft_vocab.value = 47172;
  measured.shared_wave_depth.value = 2;
  measured.draft_wave_max.value = 2;
  measured.wave_read_align.value = 2048;
  measured.depth_cost_ratio.value = 1.16;
  CalibrationKey original = Key();
  original.settings = jitllm::runtime::MeasuredWith(measured);
  const std::string text = FormatCalibration(Measured(), original);
  ASSERT_TRUE(ParseCalibration(text, original).calibration.has_value());
  for (int which = 0; which < 5; ++which) {
    SCOPED_TRACE(which);
    auto changed = measured;
    switch (which) {
      case 0:
        changed.draft_vocab.value = 16384;
        break;
      case 1:
        changed.shared_wave_depth.value = 1;
        break;
      case 2:
        changed.draft_wave_max.value = 4;
        break;
      case 3:
        changed.wave_read_align.value = 4096;
        break;
      default:
        changed.depth_cost_ratio.value = 1.2;
        break;
    }
    CalibrationKey current = original;
    current.settings = jitllm::runtime::MeasuredWith(changed);
    const auto read = ParseCalibration(text, current);
    EXPECT_FALSE(read.calibration.has_value());
    EXPECT_FALSE(read.refused);
    EXPECT_THAT(read.note, StartsWith("stale"));
  }
}

TEST(Calibration, QwenKeysEffectiveHeadRowsWithoutItsCalibratedRatio) {
  jitllm::config::ModelEntry entry;
  entry.name = "qwen";
  entry.artifact = Key().artifact;
  entry.drafter = Key().drafter;
  jitllm::runtime::ArtifactFacts facts;
  facts.architecture = "qwen4exp";
  facts.drafter_architecture = "qwen4exp-mtp";
  facts.drafter_selected_rows = 47172;
  const auto initial = jitllm::runtime::ResolveSettings(entry, facts, nullptr, false);
  ASSERT_TRUE(initial.has_value());
  CalibrationKey key = Key();
  key.settings = jitllm::runtime::MeasuredWith(*initial);
  const std::string text = FormatCalibration(Measured(), key);
  for (const std::int64_t requested : {0, 65536}) {
    entry.overrides["draft_vocab"] = requested;
    const auto uncalibrated = jitllm::runtime::ResolveSettings(entry, facts, nullptr, false);
    ASSERT_TRUE(uncalibrated.has_value());
    EXPECT_EQ(uncalibrated->draft_vocab.value, 47172U);
    CalibrationKey current = key;
    current.settings = jitllm::runtime::MeasuredWith(*uncalibrated);
    const auto read = ParseCalibration(text, current);
    ASSERT_TRUE(read.calibration.has_value()) << read.note;
    const Calibration recorded = read.calibration.value_or(Calibration{});
    const auto resolved = jitllm::runtime::ResolveSettings(entry, facts, &recorded, false);
    ASSERT_TRUE(resolved.has_value());
    EXPECT_EQ(resolved->depth_cost_ratio.value, 1.1);
    EXPECT_EQ(resolved->depth_cost_ratio.source, jitllm::runtime::SettingSource::kCalibrated);
    // Registration's next lookup resolves without calibration again.
    const auto next = jitllm::runtime::ResolveSettings(entry, facts, nullptr, false);
    ASSERT_TRUE(next.has_value());
    EXPECT_EQ(jitllm::runtime::MeasuredWith(*next), key.settings);
  }
  entry.overrides["draft_vocab"] = std::int64_t{16384};
  const auto smaller = jitllm::runtime::ResolveSettings(entry, facts, nullptr, false);
  ASSERT_TRUE(smaller.has_value());
  CalibrationKey changed = key;
  changed.settings = jitllm::runtime::MeasuredWith(*smaller);
  EXPECT_FALSE(ParseCalibration(text, changed).calibration.has_value());
  // An earlier v1 Qwen record omitting these policy dependencies is stale,
  // rather than corrupt, even when every other key component still agrees.
  CalibrationKey old = key;
  const auto policy = old.settings.find(" draft_vocab=");
  ASSERT_NE(policy, std::string::npos);
  old.settings.erase(policy);
  const auto legacy = ParseCalibration(FormatCalibration(Measured(), old), key);
  EXPECT_FALSE(legacy.calibration.has_value());
  EXPECT_FALSE(legacy.refused);
  EXPECT_THAT(legacy.note, StartsWith("stale"));
}

TEST(Calibration, ExplicitFallbackOverridesInvalidateCalibratedExecutionPolicy) {
  jitllm::config::ModelEntry entry;
  entry.name = "qwen";
  entry.artifact = Key().artifact;
  entry.drafter = Key().drafter;
  jitllm::runtime::ArtifactFacts facts;
  facts.architecture = "qwen4exp";
  facts.drafter_architecture = "qwen4exp-mtp";
  const auto fallback = jitllm::runtime::ResolveSettings(entry, facts, nullptr, false);
  ASSERT_TRUE(fallback.has_value());
  Calibration calibrated = Measured();
  calibrated.prefill_chunk = 8192;
  calibrated.max_slots = 8;
  const auto active = jitllm::runtime::ResolveSettings(entry, facts, &calibrated, false);
  ASSERT_TRUE(active.has_value());
  EXPECT_EQ(active->prefill_chunk.value, 8192U);
  EXPECT_EQ(active->max_slots.value, 8U);
  EXPECT_EQ(active->depth_cost_ratio.value, 1.1);
  CalibrationKey original = Key();
  original.settings = jitllm::runtime::MeasuredWith(*fallback);
  const std::string text = FormatCalibration(calibrated, original);
  for (const std::string name : {"prefill_chunk", "max_slots", "depth_cost_ratio"}) {
    SCOPED_TRACE(name);
    auto owner = entry;
    if (name == "prefill_chunk") {
      owner.overrides[name] = static_cast<std::int64_t>(fallback->prefill_chunk.value);
    } else if (name == "max_slots") {
      owner.overrides[name] = static_cast<std::int64_t>(fallback->max_slots.value);
    } else {
      owner.overrides[name] = fallback->depth_cost_ratio.value;
    }
    const auto uncalibrated = jitllm::runtime::ResolveSettings(owner, facts, nullptr, false);
    ASSERT_TRUE(uncalibrated.has_value());
    CalibrationKey overridden = original;
    overridden.settings = jitllm::runtime::MeasuredWith(*uncalibrated);
    const auto changed = ParseCalibration(text, overridden);
    EXPECT_FALSE(changed.calibration.has_value());
    EXPECT_FALSE(changed.refused);
    EXPECT_THAT(changed.note, StartsWith("stale"));
    // Removing an override can also enable a different calibrated policy.
    const auto removed = ParseCalibration(FormatCalibration(calibrated, overridden), original);
    EXPECT_FALSE(removed.calibration.has_value());
    EXPECT_FALSE(removed.refused);
    EXPECT_THAT(removed.note, StartsWith("stale"));
  }
}

TEST(Calibration, EffectiveContextInvalidatesContextCappedPrefillMeasurements) {
  for (const std::string architecture : {"deepseek4", "qwen4exp"}) {
    SCOPED_TRACE(architecture);
    jitllm::config::ModelEntry entry;
    entry.name = "model";
    entry.artifact = Key().artifact;
    entry.overrides["context"] = std::int64_t{1024};
    entry.overrides["prefill_chunk"] = std::int64_t{4096};
    jitllm::runtime::ArtifactFacts facts;
    facts.architecture = architecture;
    const auto before = jitllm::runtime::ResolveSettings(entry, facts, nullptr, false);
    ASSERT_TRUE(before.has_value());
    entry.overrides["context"] = std::int64_t{2048};
    const auto after = jitllm::runtime::ResolveSettings(entry, facts, nullptr, false);
    ASSERT_TRUE(after.has_value());
    EXPECT_EQ(before->prefill_chunk.value, after->prefill_chunk.value);
    EXPECT_EQ(jitllm::runtime::PrefillChunkRows(before->context.value, std::nullopt,
                                                before->prefill_chunk.value, 4096),
              1016U);
    EXPECT_EQ(jitllm::runtime::PrefillChunkRows(after->context.value, std::nullopt,
                                                after->prefill_chunk.value, 4096),
              2040U);
    CalibrationKey original = Key();
    original.settings = jitllm::runtime::MeasuredWith(*before);
    CalibrationKey current = original;
    current.settings = jitllm::runtime::MeasuredWith(*after);
    const auto read = ParseCalibration(FormatCalibration(Measured(), original), current);
    EXPECT_FALSE(read.calibration.has_value());
    EXPECT_FALSE(read.refused);
    EXPECT_THAT(read.note, StartsWith("stale"));
  }
}

TEST(Calibration, DeepSeekKeysUncalibratedWaveCostsWithoutInvalidatingItsOwnRecord) {
  jitllm::config::ModelEntry entry;
  entry.name = "ds";
  entry.artifact = Key().artifact;
  entry.drafter = Key().drafter;
  jitllm::runtime::ArtifactFacts facts;
  facts.architecture = "deepseek4";
  facts.drafter_architecture = "dflash";
  const auto initial = jitllm::runtime::ResolveSettings(entry, facts, nullptr, false);
  ASSERT_TRUE(initial.has_value());
  CalibrationKey key = Key();
  key.settings = jitllm::runtime::MeasuredWith(*initial);
  const auto read = ParseCalibration(FormatCalibration(Measured(), key), key);
  ASSERT_TRUE(read.calibration.has_value()) << read.note;
  const Calibration recorded = read.calibration.value_or(Calibration{});
  const auto active = jitllm::runtime::ResolveSettings(entry, facts, &recorded, false);
  ASSERT_TRUE(active.has_value());
  EXPECT_EQ(active->wave_costs.source, jitllm::runtime::SettingSource::kCalibrated);
  EXPECT_EQ(active->wave_costs.value[0], 2.18);
  EXPECT_EQ(active->wave_costs.value[2], 2.99);
  // Both registration and the settings command resolve without calibration
  // for their next lookup, so recording these costs preserves eligibility.
  const auto next = jitllm::runtime::ResolveSettings(entry, facts, nullptr, false);
  ASSERT_TRUE(next.has_value());
  EXPECT_EQ(jitllm::runtime::MeasuredWith(*next), key.settings);
  EXPECT_TRUE(ParseCalibration(FormatCalibration(recorded, key), key).calibration.has_value());
  entry.overrides["wave_costs"] = std::vector<double>{3.0};
  const auto changed = jitllm::runtime::ResolveSettings(entry, facts, nullptr, false);
  ASSERT_TRUE(changed.has_value());
  CalibrationKey current = key;
  current.settings = jitllm::runtime::MeasuredWith(*changed);
  const auto stale = ParseCalibration(FormatCalibration(recorded, key), current);
  EXPECT_FALSE(stale.calibration.has_value());
  EXPECT_FALSE(stale.refused);
  EXPECT_THAT(stale.note, StartsWith("stale"));
  // Hold override presence and prefix length fixed: changing only the cost
  // must also invalidate the record, independently of the override flags.
  entry.overrides["wave_costs"] = std::vector<double>{4.0};
  const auto recosted = jitllm::runtime::ResolveSettings(entry, facts, nullptr, false);
  ASSERT_TRUE(recosted.has_value());
  EXPECT_EQ(recosted->wave_costs.source, changed->wave_costs.source);
  EXPECT_EQ(recosted->wave_costs_override_count, changed->wave_costs_override_count);
  CalibrationKey cost_key = current;
  cost_key.settings = jitllm::runtime::MeasuredWith(*recosted);
  const auto changed_cost = ParseCalibration(FormatCalibration(recorded, current), cost_key);
  EXPECT_FALSE(changed_cost.calibration.has_value());
  EXPECT_FALSE(changed_cost.refused);
  EXPECT_THAT(changed_cost.note, StartsWith("stale"));
  // Earlier dependency strings remain valid v1 syntax but are not in force.
  CalibrationKey old = key;
  const auto context = old.settings.find(" context=");
  ASSERT_NE(context, std::string::npos);
  old.settings.erase(context);
  const auto legacy = ParseCalibration(FormatCalibration(recorded, old), key);
  EXPECT_FALSE(legacy.calibration.has_value());
  EXPECT_FALSE(legacy.refused);
  EXPECT_THAT(legacy.note, StartsWith("stale"));
}

TEST(Calibration, ExplicitFallbackWaveCostsInvalidateInBothDirections) {
  jitllm::config::ModelEntry entry;
  entry.name = "ds";
  entry.artifact = Key().artifact;
  entry.drafter = Key().drafter;
  jitllm::runtime::ArtifactFacts facts;
  facts.architecture = "deepseek4";
  facts.drafter_architecture = "dflash";
  const auto fallback = jitllm::runtime::ResolveSettings(entry, facts, nullptr, false);
  ASSERT_TRUE(fallback.has_value());
  CalibrationKey original = Key();
  original.settings = jitllm::runtime::MeasuredWith(*fallback);
  entry.overrides["wave_costs"] = std::vector<double>{fallback->wave_costs.value[0]};
  const auto owner = jitllm::runtime::ResolveSettings(entry, facts, nullptr, false);
  ASSERT_TRUE(owner.has_value());
  EXPECT_EQ(owner->wave_costs.value, fallback->wave_costs.value);
  CalibrationKey overridden = original;
  overridden.settings = jitllm::runtime::MeasuredWith(*owner);
  EXPECT_NE(overridden.settings, original.settings);
  for (const bool removed : {false, true}) {
    SCOPED_TRACE(removed);
    const auto read =
        ParseCalibration(FormatCalibration(Measured(), removed ? overridden : original),
                         removed ? original : overridden);
    EXPECT_FALSE(read.calibration.has_value());
    EXPECT_FALSE(read.refused);
    EXPECT_THAT(read.note, StartsWith("stale"));
  }
}

TEST(Calibration, EqualWaveCostVectorsWithDifferentOverridePrefixesHaveDifferentKeys) {
  jitllm::config::ModelEntry entry;
  entry.name = "ds";
  entry.artifact = Key().artifact;
  entry.drafter = Key().drafter;
  jitllm::runtime::ArtifactFacts facts;
  facts.architecture = "deepseek4";
  facts.drafter_architecture = "dflash";
  entry.overrides["wave_costs"] = std::vector<double>{jitllm::runtime::kDsv4WaveCosts[0]};
  const auto one = jitllm::runtime::ResolveSettings(entry, facts, nullptr, false);
  ASSERT_TRUE(one.has_value());
  const auto first_entry = entry;
  entry.overrides["wave_costs"] =
      std::vector<double>{jitllm::runtime::kDsv4WaveCosts[0], jitllm::runtime::kDsv4WaveCosts[1]};
  const auto two = jitllm::runtime::ResolveSettings(entry, facts, nullptr, false);
  ASSERT_TRUE(two.has_value());
  EXPECT_EQ(one->wave_costs.value, two->wave_costs.value);
  EXPECT_EQ(one->wave_costs.source, jitllm::runtime::SettingSource::kOverride);
  EXPECT_EQ(two->wave_costs.source, jitllm::runtime::SettingSource::kOverride);
  EXPECT_EQ(one->wave_costs_override_count, 1U);
  EXPECT_EQ(two->wave_costs_override_count, 2U);
  Calibration measured;
  measured.wave_costs[1] = 4.5;
  const auto first_active = jitllm::runtime::ResolveSettings(first_entry, facts, &measured, false);
  const auto second_active = jitllm::runtime::ResolveSettings(entry, facts, &measured, false);
  ASSERT_TRUE(first_active.has_value());
  ASSERT_TRUE(second_active.has_value());
  EXPECT_EQ(first_active->wave_costs.value[1], 4.5);
  EXPECT_EQ(second_active->wave_costs.value[1], jitllm::runtime::kDsv4WaveCosts[1]);
  CalibrationKey first = Key();
  first.settings = jitllm::runtime::MeasuredWith(*one);
  CalibrationKey second = first;
  second.settings = jitllm::runtime::MeasuredWith(*two);
  EXPECT_NE(first.settings, second.settings);
  for (const bool shortened : {false, true}) {
    SCOPED_TRACE(shortened);
    const auto read = ParseCalibration(FormatCalibration(measured, shortened ? second : first),
                                       shortened ? first : second);
    EXPECT_FALSE(read.calibration.has_value());
    EXPECT_FALSE(read.refused);
    EXPECT_THAT(read.note, StartsWith("stale"));
  }
}

TEST(Calibration, QwenDoesNotKeyIgnoredDeepSeekWaveCostOverrides) {
  jitllm::config::ModelEntry entry;
  entry.name = "qwen";
  entry.artifact = Key().artifact;
  jitllm::runtime::ArtifactFacts facts;
  facts.architecture = "qwen4exp";
  const auto before = jitllm::runtime::ResolveSettings(entry, facts, nullptr, false);
  ASSERT_TRUE(before.has_value());
  entry.overrides["wave_costs"] = std::vector<double>{3.0};
  const auto after = jitllm::runtime::ResolveSettings(entry, facts, nullptr, false);
  ASSERT_TRUE(after.has_value());
  EXPECT_FALSE(after->ignored.empty());
  EXPECT_EQ(jitllm::runtime::MeasuredWith(*before), jitllm::runtime::MeasuredWith(*after));
}

// Measured for another device, driver, build, drafter or measured-with
// settings: not used, not an error (it is measured again).
TEST(Calibration, AStaleRecordIsNotUsed) {
  const std::string text = FormatCalibration(Measured(), Key());
  for (int which = 0; which < 4; ++which) {
    CalibrationKey now = Key();
    if (which == 0) {
      now.device = "another device";
    } else if (which == 1) {
      now.build = "another build";
    } else if (which == 2) {
      now.drafter = std::string(64, 'c');
    } else {
      now.settings =
          "speculation=false draft_rows=3 prefill_chunk=4096 max_slots=4 "
          "prefill_outa_hca=true/true wave_form=auto";
    }
    const auto read = ParseCalibration(text, now);
    EXPECT_FALSE(read.calibration.has_value());
    EXPECT_FALSE(read.refused);
    EXPECT_THAT(read.note, StartsWith("stale"));
  }
}

// Not this format, another artifact's, or a value it may not hold: refused.
TEST(Calibration, ACorruptOrForeignRecordIsRefused) {
  const std::string good = FormatCalibration(Measured(), Key());
  CalibrationKey other = Key();
  other.artifact = std::string(64, 'f');
  const auto foreign = ParseCalibration(good, other);
  EXPECT_FALSE(foreign.calibration.has_value());
  EXPECT_TRUE(foreign.refused);
  EXPECT_THAT(foreign.note, HasSubstr("artifact"));

  const std::string head = std::format(
      R"({{"format":"jitllm-model-calibration-v1","artifact":"{}","drafter":"{}","device":"{}","build":"{}","settings":"{}",)",
      Key().artifact, Key().drafter, Key().device, Key().build, Key().settings);
  // The head itself is valid.
  EXPECT_TRUE(ParseCalibration(head + R"("values":{}})", Key()).calibration.has_value());
  for (const std::string& bad : {
           std::string("not json"),
           std::string("[]"),
           good.substr(0, good.size() - 1),
           std::string(R"({"format":"jitllm-model-calibration-v2"})"),
           head + R"("values":{}, "extra":1})",
           head + R"("values":{"context":4096}})",
           head + R"("values":{"prefill_floor_tok_s":0}})",
           head + R"("values":{"prefill_floor_tok_s":1.5}})",
           head + R"("values":{"decode_floor_tok_s":"5"}})",
           head + R"("values":{"recompute_ms_per_token":0}})",
           head + R"("values":{"recompute_seconds_per_gib":64}})",
           head + R"("values":{"depth_cost_ratio":-1}})",
           head + R"("values":{"wave_costs":[]}})",
           head + R"("values":{"wave_costs":[1,1,1,1,1,1,1,1]}})",
           head + R"("values":{"wave_costs":[-1]}})",
           head + R"("values":{"wave_costs":["2"]}})",
           head + R"("values":{"max_slots":17}})",
           head + R"("values":[]})",
       }) {
    const auto read = ParseCalibration(bad, Key());
    EXPECT_FALSE(read.calibration.has_value()) << bad;
    EXPECT_TRUE(read.refused) << bad;
    EXPECT_THAT(read.note, StartsWith("refused")) << bad;
  }
}

fs::path Scratch() {
  const char* scratch = std::getenv("JITLLM_TEST_SCRATCH");  // NOLINT(concurrency-mt-unsafe)
  return scratch != nullptr ? fs::path(scratch) : fs::path(::testing::TempDir());
}

// Written atomically under the state role, 0600 in a 0700 directory, and
// read back under the trust rules; a record others could change is refused.
TEST(Calibration, WritesAndReadsItsRecord) {
  std::error_code error;
  fs::create_directories(Scratch(), error);
  std::string pattern = (Scratch() / "calibration-XXXXXX").string();
  ASSERT_NE(::mkdtemp(pattern.data()), nullptr);
  const fs::path state = pattern;
  const uid_t self = ::geteuid();
  auto none = jitllm::runtime::ReadCalibration(state, Key(), self);
  EXPECT_FALSE(none.calibration.has_value());
  EXPECT_FALSE(none.refused);
  EXPECT_EQ(none.note, "none recorded");

  ASSERT_TRUE(jitllm::runtime::WriteCalibration(state, Key(), Measured()).has_value());
  const fs::path file = jitllm::runtime::CalibrationPath(state, Key().artifact);
  struct stat status{};
  ASSERT_EQ(::stat(file.c_str(), &status), 0);
  EXPECT_EQ(status.st_mode & 0777U, 0600U);
  ASSERT_EQ(::stat(file.parent_path().c_str(), &status), 0);
  EXPECT_EQ(status.st_mode & 0077U, 0U);
  auto read = jitllm::runtime::ReadCalibration(state, Key(), self);
  ASSERT_TRUE(read.calibration.has_value()) << read.note;
  EXPECT_EQ(read.calibration.value_or(Calibration{}).decode_floor_tok_s, 7U);

  // Replaced whole; no temporary file stays.
  Calibration more = Measured();
  more.max_slots = 6;
  ASSERT_TRUE(jitllm::runtime::WriteCalibration(state, Key(), more).has_value());
  read = jitllm::runtime::ReadCalibration(state, Key(), self);
  ASSERT_TRUE(read.calibration.has_value()) << read.note;
  EXPECT_EQ(read.calibration.value_or(Calibration{}).max_slots, 6U);
  std::size_t entries = 0;
  for (const auto& entry : fs::directory_iterator(file.parent_path())) {
    (void)entry;
    ++entries;
  }
  EXPECT_EQ(entries, 1U);

  ASSERT_EQ(::chmod(file.c_str(), 0666), 0);
  read = jitllm::runtime::ReadCalibration(state, Key(), self);
  EXPECT_FALSE(read.calibration.has_value());
  EXPECT_TRUE(read.refused);
  EXPECT_THAT(read.note, HasSubstr("other users can write it"));
  fs::remove_all(state, error);
}

// A model's first uses, measured: each value once enough samples agree,
// short context only; a value known already is not measured again.
TEST(Calibration, MeasuresFirstUses) {
  CalibrationSamples samples;
  EXPECT_FALSE(samples.Take(Calibration{}).has_value());
  for (std::size_t i = 0; i + 1 < CalibrationSamples::kSamples; ++i) {
    samples.PrefillChunk(0, 4096, 4.0);  // 1,024 tokens a second
    samples.DecodeStep(100, 2.5, 0.1);   // 25
    samples.Wave(2, true, 0.15);
    samples.Wave(2, false, 0.075);
    samples.DraftStep(3, 0.058);
    samples.DraftStep(2, 0.05);
  }
  // Long context, too few rows, nothing committed, widths out of range,
  // short whole prefills: not counted.
  samples.PrefillChunk(CalibrationSamples::kShortContext, 4096, 100.0);
  samples.PrefillChunk(0, 8, 100.0);
  samples.DecodeStep(CalibrationSamples::kShortContext, 1.0, 100.0);
  samples.DecodeStep(0, 0.0, 100.0);
  samples.Wave(1, true, 9.0);
  samples.Wave(9, true, 9.0);
  samples.DraftStep(4, 9.0);
  for (int i = 0; i < 3; ++i) {
    samples.Prefill(4096, 1.0);
  }
  EXPECT_FALSE(samples.Take(Calibration{}).has_value());

  samples.PrefillChunk(4096, 4096, 4.0);
  samples.DecodeStep(100, 2.5, 0.1);
  samples.Wave(2, true, 0.15);
  samples.Wave(2, false, 0.075);
  samples.DraftStep(3, 0.058);
  samples.DraftStep(2, 0.05);
  // The recompute cost a token: the slowest of the whole prefills (the
  // longest here), not their median.
  samples.Prefill(8192, 8.0);    // 0.98 ms a token
  samples.Prefill(16384, 22.9);  // 1.40
  samples.Prefill(9000, 9.0);    // 1.00
  const auto taken = samples.Take(Calibration{});
  ASSERT_TRUE(taken.has_value());
  const Calibration c = taken.value_or(Calibration{});
  EXPECT_EQ(c.prefill_floor_tok_s, 341U);
  EXPECT_EQ(c.decode_floor_tok_s, 8U);
  EXPECT_NEAR(c.recompute_ms_per_token.value_or(0), 22.9 * 1000 / 16384, 1e-9);
  EXPECT_NEAR(c.wave_costs[0].value_or(0), 2.0, 1e-9);
  EXPECT_FALSE(c.wave_costs[1].has_value());
  EXPECT_NEAR(c.depth_cost_ratio.value_or(0), 1.16, 1e-9);
  EXPECT_FALSE(c.prefill_chunk.has_value());
  EXPECT_FALSE(c.max_slots.has_value());
  // Nothing new since, or nothing it does not know: no record to write.
  EXPECT_FALSE(samples.Take(c).has_value());
  samples.DecodeStep(100, 1.0, 1.0);
  EXPECT_FALSE(samples.Take(c).has_value());
  // A record known before measures only what it lacks.
  Calibration known;
  known.decode_floor_tok_s = 3;
  samples.DecodeStep(100, 1.0, 1.0);
  const auto merged = samples.Take(known);
  ASSERT_TRUE(merged.has_value());
  EXPECT_EQ(merged.value_or(Calibration{}).decode_floor_tok_s, 3U);
  EXPECT_EQ(merged.value_or(Calibration{}).prefill_floor_tok_s, 341U);
}

using WaveMode = jitllm::execution::AdaptiveWaveMode;
using WaveChoice = WaveMode::Mode;
using WaveExploration = jitllm::runtime::WaveCostExploration;

// The real counted policy, fed the resolver's costs; exploration receives
// its selected mode exactly as the DeepSeek runner does.
WaveMode CountedWaveMode(const jitllm::runtime::ModelSettings& settings) {
  WaveMode::Costs costs{};
  for (std::size_t i = 0; i < settings.wave_costs.value.size(); ++i) {
    costs[i + 2] = settings.wave_costs.value[i];
  }
  WaveMode::Force force = WaveMode::Force::kNone;
  if (settings.wave_form.value == jitllm::config::WaveForm::kPlain) {
    force = WaveMode::Force::kPlain;
  } else if (settings.wave_form.value == jitllm::config::WaveForm::kSpeculative) {
    force = WaveMode::Force::kSpeculative;
  }
  return WaveMode(costs, force);
}

TEST(Calibration, ExplicitWaveCostsKeepZeroAndCountedChoicesThroughExploration) {
  jitllm::config::ModelEntry entry;
  entry.name = "ds";
  entry.overrides["wave_costs"] = std::vector<double>{0, 2.08};
  jitllm::runtime::ArtifactFacts facts;
  facts.architecture = "deepseek4";
  facts.drafter_architecture = "dflash";
  const auto resolved = jitllm::runtime::ResolveSettings(entry, facts, nullptr, false);
  ASSERT_TRUE(resolved.has_value());
  EXPECT_EQ(resolved->wave_costs_override_count, 2U);
  WaveMode zero_counted = CountedWaveMode(*resolved);
  WaveMode finite_counted = CountedWaveMode(*resolved);
  WaveExploration explore;
  for (std::size_t i = 0; i < WaveExploration::kWaves + 8; ++i) {
    SCOPED_TRACE(i);
    // Zero always speculates even when one kept token would never pay a
    // finite cost. The second supplied cost pays three kept tokens and
    // also stays speculative, including odd exploratory wave numbers.
    const auto zero = explore.Choose(zero_counted.Choose(2), 2, false, *resolved, Calibration{});
    EXPECT_EQ(zero, WaveChoice::kSpeculative);
    zero_counted.Observe(2, zero, 2, 2);
    const auto finite =
        explore.Choose(finite_counted.Choose(3), 3, false, *resolved, Calibration{});
    EXPECT_EQ(finite, WaveChoice::kSpeculative);
    finite_counted.Observe(3, finite, 9, 3);
  }
  // A finite override remains adaptive rather than forcing speculation:
  // three full verifies keeping only their anchor switch to plain.
  WaveMode low_acceptance = CountedWaveMode(*resolved);
  WaveExploration low_explore;
  for (int i = 0; i < 3; ++i) {
    const auto selected =
        low_explore.Choose(low_acceptance.Choose(3), 3, false, *resolved, Calibration{});
    EXPECT_EQ(selected, WaveChoice::kSpeculative);
    low_acceptance.Observe(3, selected, 3, 3);
  }
  EXPECT_EQ(low_explore.Choose(low_acceptance.Choose(3), 3, false, *resolved, Calibration{}),
            WaveChoice::kPlain);
}

TEST(Calibration, ExplicitPrefixLeavesOnlyUnknownSuffixWidthsExploring) {
  jitllm::config::ModelEntry entry;
  entry.name = "ds";
  entry.overrides["wave_costs"] = std::vector<double>{0};
  jitllm::runtime::ArtifactFacts facts;
  facts.architecture = "deepseek4";
  facts.drafter_architecture = "dflash";
  const auto resolved = jitllm::runtime::ResolveSettings(entry, facts, nullptr, false);
  ASSERT_TRUE(resolved.has_value());
  WaveMode counted = CountedWaveMode(*resolved);
  WaveExploration explore;
  // Width two's override does not consume width three's exploration.
  for (std::size_t i = 0; i < WaveExploration::kWaves; ++i) {
    EXPECT_EQ(explore.Choose(counted.Choose(2), 2, false, *resolved, Calibration{}),
              WaveChoice::kSpeculative);
    const auto selected = explore.Choose(counted.Choose(3), 3, false, *resolved, Calibration{});
    EXPECT_EQ(selected, i % 2 == 0 ? WaveChoice::kSpeculative : WaveChoice::kPlain);
    counted.Observe(3, selected, 3, 3);
  }
  EXPECT_EQ(explore.Choose(counted.Choose(3), 3, false, *resolved, Calibration{}),
            WaveChoice::kPlain);  // bounded exploration ends; low acceptance decides

  Calibration known;
  known.wave_costs[0] = 1000;  // an explicit prefix still replaces a calibrated value
  known.wave_costs[1] = 1.5;
  const auto calibrated = jitllm::runtime::ResolveSettings(entry, facts, &known, false);
  ASSERT_TRUE(calibrated.has_value());
  EXPECT_EQ(calibrated->wave_costs.value[0], 0);
  EXPECT_EQ(calibrated->wave_costs.value[1], 1.5);
  WaveMode paid = CountedWaveMode(*calibrated);
  WaveExploration measured;
  for (std::size_t i = 0; i < WaveExploration::kWaves; ++i) {
    const auto selected = measured.Choose(paid.Choose(3), 3, false, *calibrated, known);
    EXPECT_EQ(selected, WaveChoice::kSpeculative);
    paid.Observe(3, selected, 9, 3);
  }
}

TEST(Calibration, ForcedAndSampledWavesDoNotConsumeCalibrationExploration) {
  jitllm::config::ModelEntry entry;
  entry.name = "ds";
  jitllm::runtime::ArtifactFacts facts;
  facts.architecture = "deepseek4";
  facts.drafter_architecture = "dflash";
  const auto automatic = jitllm::runtime::ResolveSettings(entry, facts, nullptr, false);
  ASSERT_TRUE(automatic.has_value());
  WaveMode counted = CountedWaveMode(*automatic);
  WaveExploration explore;
  for (std::size_t i = 0; i < WaveExploration::kWaves; ++i) {
    EXPECT_EQ(explore.Choose(counted.Choose(2, true), 2, true, *automatic, Calibration{}),
              WaveChoice::kSpeculative);
  }
  for (const std::string form : {"plain", "speculative"}) {
    entry.overrides["wave_form"] = form;
    const auto forced = jitllm::runtime::ResolveSettings(entry, facts, nullptr, false);
    ASSERT_TRUE(forced.has_value());
    WaveMode policy = CountedWaveMode(*forced);
    const auto expected = form == "plain" ? WaveChoice::kPlain : WaveChoice::kSpeculative;
    for (std::size_t i = 0; i < WaveExploration::kWaves; ++i) {
      for (const bool sampled : {false, true}) {
        EXPECT_EQ(explore.Choose(policy.Choose(2, sampled), 2, sampled, *forced, Calibration{}),
                  expected);
      }
    }
  }
  // Those bypassed waves leave this width's first unknown-cost samples
  // untouched, and a count-zero profile retains the existing alternation.
  for (std::size_t i = 0; i < WaveExploration::kWaves; ++i) {
    EXPECT_EQ(explore.Choose(counted.Choose(2), 2, false, *automatic, Calibration{}),
              i % 2 == 0 ? WaveChoice::kSpeculative : WaveChoice::kPlain);
  }
}

TEST(Calibration, FullWaveCostPrefixIncludesWidthEightAndInvalidWidthsStayUnchanged) {
  jitllm::config::ModelEntry entry;
  entry.name = "ds";
  entry.overrides["wave_costs"] = std::vector<double>(7, 0);
  jitllm::runtime::ArtifactFacts facts;
  facts.architecture = "deepseek4";
  facts.drafter_architecture = "dflash";
  const auto all = jitllm::runtime::ResolveSettings(entry, facts, nullptr, false);
  ASSERT_TRUE(all.has_value());
  EXPECT_EQ(all->wave_costs_override_count, 7U);
  WaveMode counted = CountedWaveMode(*all);
  WaveExploration explore;
  for (std::size_t i = 0; i < WaveExploration::kWaves + 1; ++i) {
    for (std::uint32_t width = 2; width <= WaveMode::kMaxWidth; ++width) {
      EXPECT_EQ(explore.Choose(counted.Choose(width), width, false, *all, Calibration{}),
                WaveChoice::kSpeculative);
    }
  }
  for (const std::uint32_t width : {0U, 1U, 9U, 1000U}) {
    EXPECT_EQ(explore.Choose(WaveChoice::kPlain, width, false, *all, Calibration{}),
              WaveChoice::kPlain);
  }
}

}  // namespace
