// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "qwen38_vocab.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <format>
#include <string>
#include <string_view>
#include <vector>

namespace {
namespace dv = llmp::benchmarks::draft_vocab;

std::vector<dv::Example> Examples() {
  std::vector<dv::Example> out;
  for (std::string_view domain : {"code", "prose", "instruction", "arithmetic"}) {
    for (std::uint32_t rows : {8192U, 32768U}) {
      out.push_back({.id = std::format("cal-{}-{}", domain, rows),
                     .domain = std::string(domain),
                     .split = "calibration",
                     .nominal_tokens = rows,
                     .stable_boundary = rows - 4,
                     .prompt = std::vector<std::int32_t>(rows, 13),
                     .continuation = std::vector<std::int32_t>(32, 14),
                     .anchors = {0, 8, 16, 24}});
    }
  }
  return out;
}

TEST(Qwen38Vocab, RequiresCompleteIndependentCellsAndBoundedFrozenAnchors) {
  auto examples = Examples();
  ASSERT_TRUE(dv::CheckExamples(examples, dv::kContext, 248320));
  examples.pop_back();
  EXPECT_FALSE(dv::CheckExamples(examples, dv::kContext, 248320));
  examples = Examples();
  examples[0].split = "held_out";
  EXPECT_FALSE(dv::CheckExamples(examples, dv::kContext, 248320));
  examples = Examples();
  examples.back().domain = "code";
  EXPECT_FALSE(dv::CheckExamples(examples, dv::kContext, 248320));
  examples = Examples();
  examples[0].id = "../escape";
  EXPECT_FALSE(dv::CheckExamples(examples, dv::kContext, 248320));
  examples = Examples();
  examples[0].anchors = {0, 8, 8, 24};
  EXPECT_FALSE(dv::CheckExamples(examples, dv::kContext, 248320));
  examples = Examples();
  examples[0].continuation.resize(27);
  EXPECT_FALSE(dv::CheckExamples(examples, dv::kContext, 248320));
}

TEST(Qwen38Vocab, RefusesChangedBudgetBoundaryAndOutOfRangeOrOversizedInputs) {
  auto examples = Examples();
  EXPECT_FALSE(dv::CheckExamples(examples, dv::kContext + 1, 248320));
  EXPECT_FALSE(dv::CheckExamples(examples, 32768, 248320));
  examples[0].prompt.resize(8192 - 129);
  examples[0].stable_boundary = 1;
  EXPECT_FALSE(dv::CheckExamples(examples, dv::kContext, 248320));
  examples = Examples();
  examples[0].stable_boundary = 8192;
  EXPECT_FALSE(dv::CheckExamples(examples, dv::kContext, 248320));
  examples = Examples();
  examples[0].continuation[4] = 248320;
  EXPECT_FALSE(dv::CheckExamples(examples, dv::kContext, 248320));
  examples[0].continuation[4] = -1;
  EXPECT_FALSE(dv::CheckExamples(examples, dv::kContext, 248320));
  examples = Examples();
  examples[0].continuation.resize(257);
  EXPECT_FALSE(dv::CheckExamples(examples, dv::kContext, 248320));
}

std::string Prepared() {
  std::string records;
  for (const auto& example : Examples()) {
    std::string ids;
    for (std::size_t i = 0; i < example.prompt.size(); ++i) {
      ids += i == 0 ? "13" : ",13";
    }
    std::string continuation;
    for (std::size_t i = 0; i < example.continuation.size(); ++i) {
      continuation += i == 0 ? "14" : ",14";
    }
    records += std::format(
        R"({}{{"id":"{}","domain":"{}","nominal_tokens":{},"stable_boundary":{},"prompt_ids":[{}],"continuation_ids":[{}],"anchors":[0,8,16,24]}})",
        records.empty() ? "" : ",", example.id, example.domain, example.nominal_tokens,
        example.stable_boundary, ids, continuation);
  }
  return std::format(
      R"({{"format":"llmp-qwen-own-vocab-prepared-v1","split":"calibration","context_capacity":33792,"prefill_chunk":8192,"draft_depth":3,"examples":[{}],"target_signal":"native_control_natural_argmax"}})",
      records);
}

TEST(Qwen38Vocab, ParsesCompleteFrozenIdsAndRefusesFractionalOrUnlabelledIdsBeforeSetup) {
  auto text = Prepared();
  const auto parsed = dv::ReadExamples(text, dv::kContext, 248320);
  ASSERT_TRUE(parsed);
  EXPECT_EQ(parsed->size(), 8);
  EXPECT_EQ(parsed->front().split, "calibration");
  EXPECT_EQ(parsed->front().prompt.size(), 8192);
  EXPECT_EQ(parsed->back().continuation.size(), 32);
  const auto position = text.find("\"prompt_ids\":[13");
  ASSERT_NE(position, std::string::npos);
  text.replace(position + std::string_view("\"prompt_ids\":[").size(), 2, "13.5");
  EXPECT_FALSE(dv::ReadExamples(text, dv::kContext, 248320));
  EXPECT_FALSE(dv::ReadExamples(Prepared(), dv::kContext - 1, 248320));
  auto changed = Prepared();
  const auto chunk = changed.find("\"prefill_chunk\":8192");
  ASSERT_NE(chunk, std::string::npos);
  changed.replace(chunk, std::string_view("\"prefill_chunk\":8192").size(),
                  "\"prefill_chunk\":4096");
  EXPECT_FALSE(dv::ReadExamples(changed, dv::kContext, 248320));
  EXPECT_FALSE(dv::ReadExamples("{}", dv::kContext, 248320));
  EXPECT_FALSE(dv::ReadExamples(
      R"({"format":"llmp-qwen-own-vocab-prepared-v1","split":"calibration","examples":[],"target_signal":"authored_answer"})",
      dv::kContext, 248320));
}

}  // namespace
