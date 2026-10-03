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
              "prefill_outa_hca=true/true wave_form=auto"};
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

}  // namespace
