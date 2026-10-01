// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "long_context_tasks.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "expected_error.h"

namespace {
namespace lc = jitllm::benchmarks::long_context;
using jitllm::test_support::Failed;

// Independent byte-token codec makes the complete final positions
// observable, including template bytes; it is no model-tokenizer claim.
class Bytes final : public lc::Codec {
 public:
  std::expected<lc::Encoding, std::string> Encode(std::string_view user) const override {
    lc::Encoding e;
    e.rendered = "T\n" + std::string(user) + "G\n";
    for (std::size_t i = 0; i < e.rendered.size(); ++i) {
      e.ids.push_back(static_cast<unsigned char>(e.rendered[i]));
      e.byte_ends.push_back(i + 1);
    }
    return e;
  }
};
lc::Blueprint Small() {
  return {"small",
          2048,
          4096,
          128,
          lc::AnswerKind::kNumbers,
          {{"Record k004201 = 731942.\n", 512}},
          "Return exactly the value assigned to k004201 as a JSON array.",
          {"731942"},
          {}};
}

TEST(LongContextTasks, FixedCasesAndDynamicUsedExtentCounts) {
  const auto cases = lc::FixedCases();
  ASSERT_EQ(cases.size(), 4U);
  EXPECT_EQ(cases[0].prompt_tokens, 30720U);
  EXPECT_EQ(cases[1].prompt_tokens, 126976U);
  EXPECT_EQ(cases[2].facts[2].target_token, 57344U);
  EXPECT_EQ(cases[3].kind, lc::AnswerKind::kQualitative);
  EXPECT_EQ(cases[3].output_tokens, 768U);
  const auto count = [](std::uint32_t tokens) {
    auto result = lc::HcaCalls(tokens);
    EXPECT_TRUE(result.has_value());
    return result.value_or(0);
  };
  EXPECT_EQ(count(30720), 300U);
  EXPECT_EQ(count(32768), 320U);
  EXPECT_EQ(count(65536), 320U);  // 512 cells remain ordinary
  EXPECT_EQ(count(98304), 320U);  // 768 cells remain ordinary
  EXPECT_EQ(count(100352), 340U);
  EXPECT_EQ(count(126976), 600U);
  EXPECT_EQ(count(131072), 640U);
  EXPECT_FALSE(lc::HcaCalls(0));
  EXPECT_FALSE(lc::HcaCalls(30721));
  EXPECT_FALSE(lc::HcaCalls(133120));
}

TEST(LongContextTasks, ExactRenderedPlacementPreservesFactsAndQuestion) {
  Bytes codec;
  const auto b = Small();
  auto prepared = lc::Prepare(b, codec);
  ASSERT_TRUE(prepared) << Failed(prepared).value_or("");
  EXPECT_EQ(prepared->encoded.ids.size(), b.prompt_tokens);
  ASSERT_EQ(prepared->spans.size(), 1U);
  const auto& span = prepared->spans.front();
  EXPECT_EQ(span.first_token, 512U);
  EXPECT_EQ(span.first_byte, 512U);
  EXPECT_EQ(prepared->encoded.rendered.substr(span.first_byte, span.end_byte - span.first_byte),
            b.facts.front().text);
  EXPECT_NE(prepared->user.find(b.question), std::string::npos);
  EXPECT_EQ(prepared->user.find(b.facts.front().text), prepared->user.rfind(b.facts.front().text));
  EXPECT_LE(prepared->encoding_calls, 256U);
  auto repeated = lc::Prepare(b, codec);
  ASSERT_TRUE(repeated) << Failed(repeated).value_or("");
  EXPECT_EQ(repeated->encoded.ids, prepared->encoded.ids);
  EXPECT_EQ(repeated->user, prepared->user);
}

TEST(LongContextTasks, FactTerminatorRemainsACompleteTokenBeforeQuestion) {
  // Independent tokenizer witness: adjacent newlines are one token, so a
  // fact's final newline cannot be split from an extra separator newline.
  class Newlines final : public lc::Codec {
   public:
    std::expected<lc::Encoding, std::string> Encode(std::string_view user) const override {
      lc::Encoding e;
      e.rendered = "T\n" + std::string(user) + "G\n";
      for (std::size_t i = 0; i < e.rendered.size();) {
        const bool paired =
            e.rendered[i] == '\n' && i + 1 < e.rendered.size() && e.rendered[i + 1] == '\n';
        e.ids.push_back(paired ? 256 : static_cast<unsigned char>(e.rendered[i]));
        i += paired ? std::size_t{2} : std::size_t{1};
        e.byte_ends.push_back(i);
      }
      return e;
    }
  } codec;
  const auto b = Small();
  auto prepared = lc::Prepare(b, codec);
  ASSERT_TRUE(prepared) << Failed(prepared).value_or("");
  ASSERT_EQ(prepared->spans.size(), 1U);
  const auto& span = prepared->spans.front();
  EXPECT_EQ(span.first_token, b.facts.front().target_token);
  EXPECT_EQ(prepared->encoded.byte_ends[span.end_token - 1], span.end_byte);
  EXPECT_EQ(prepared->encoded.ids.size(), b.prompt_tokens);
  EXPECT_EQ(prepared->user.find(b.facts.front().text), prepared->user.rfind(b.facts.front().text));
  EXPECT_NE(prepared->user.find("Question: " + b.question), std::string::npos);
  EXPECT_EQ(prepared->user.find("\n\nQuestion:"), std::string::npos);
}

TEST(LongContextTasks, RefusesMutationExhaustionAndUnreachableBoundaries) {
  Bytes codec;
  auto b = Small();
  b.answers = {"731943"};
  EXPECT_FALSE(lc::Prepare(b, codec));
  b = Small();
  b.facts.push_back(b.facts.front());
  b.facts.back().target_token = 1024;
  EXPECT_FALSE(lc::Prepare(b, codec));
  b = Small();
  b.facts.front().target_token = 1;
  EXPECT_FALSE(lc::Prepare(b, codec));
  EXPECT_FALSE(lc::Prepare(Small(), codec, {.max_encoding_calls = 1}));
  EXPECT_FALSE(lc::Prepare(Small(), codec, {.max_bytes = 64}));
  class Broken final : public lc::Codec {
   public:
    std::expected<lc::Encoding, std::string> Encode(std::string_view /*user*/) const override {
      return lc::Encoding{"x", {10, 11}, {1, 1}};
    }
  } broken;
  EXPECT_FALSE(lc::Prepare(Small(), broken));
}

TEST(LongContextTasks, ChronologyRejectsUnboundOrExtraVariables) {
  Bytes codec;
  auto b = Small();
  b.kind = lc::AnswerKind::kVariables;
  b.facts = {{"Binding a = 123456.\n", 512}, {"Binding b = a.\n", 1024}};
  b.answers = {"a", "b"};
  auto prepared = lc::Prepare(b, codec);
  ASSERT_TRUE(prepared) << Failed(prepared).value_or("");
  b.facts.back().text = "Binding b = absent.\n";
  EXPECT_FALSE(lc::Prepare(b, codec));
  b.facts.back().text = "Binding b = a.\n";
  b.answers.emplace_back("noise");
  EXPECT_FALSE(lc::Prepare(b, codec));
}

TEST(LongContextTasks, ExactAnswerRejectsSubstringRecallAndTruncation) {
  const auto cases = lc::FixedCases();
  const auto accepted = [&](std::size_t i, std::string_view answer, std::string_view stop = "eos") {
    auto result = lc::ExactAnswer(cases[i], answer, stop);
    EXPECT_TRUE(result.has_value());
    return result.value_or(false);
  };
  EXPECT_TRUE(accepted(0, " [731942] \n"));
  EXPECT_FALSE(accepted(0, "The answer is731942."));
  EXPECT_FALSE(accepted(0, "[731942,0]"));
  EXPECT_FALSE(accepted(0, "[\"731942\"]"));
  EXPECT_FALSE(accepted(0, "[731942.0]"));
  EXPECT_FALSE(accepted(0, "[731942] trailing"));
  EXPECT_FALSE(accepted(0, "[731942]", "length"));
  EXPECT_TRUE(accepted(1, "[574829,390461,815207,620143]"));
  EXPECT_FALSE(accepted(1, "[620143,815207,390461,390461]"));
  EXPECT_TRUE(accepted(2, "[\"v_hop42d\",\"v_hop42b\",\"v_seed42\",\"v_hop42a\",\"v_hop42c\"]"));
  EXPECT_FALSE(accepted(2, "[\"v_hop42d\",\"v_hop42b\",\"v_seed42\",\"v_hop42a\"]"));
  EXPECT_FALSE(lc::ExactAnswer(cases[3], "valid-looking code", "eos"));
}

TEST(LongContextTasks, StopsAreExplicitUniqueAndIncludeTheFirstToken) {
  auto stops = lc::ParseStopIds("2,129279", 129280);
  ASSERT_TRUE(stops) << Failed(stops).value_or("");
  EXPECT_TRUE(lc::IsStop(*stops, 2));
  EXPECT_TRUE(lc::IsStop(*stops, 129279));
  EXPECT_FALSE(lc::IsStop(*stops, 1));
  EXPECT_FALSE(lc::IsStop({}, 2));
  for (std::string_view text : {"", "2,2", "2,", ",2", "-1", "129280", "2junk", "2, 3"})
    EXPECT_FALSE(lc::ParseStopIds(text, 129280));
  EXPECT_FALSE(lc::ParseStopIds("2147483648", std::numeric_limits<std::uint32_t>::max()));
  auto largest = lc::ParseStopIds("2147483647", std::numeric_limits<std::uint32_t>::max());
  ASSERT_TRUE(largest) << Failed(largest).value_or("");
  EXPECT_EQ(largest->front(), std::numeric_limits<std::int32_t>::max());
  const std::vector<std::int32_t> first_eos{2};
  EXPECT_TRUE(lc::IsStop(*stops, first_eos.back()));
  const std::vector<std::int32_t> body_then_eos{10, 11, 129279};
  EXPECT_TRUE(lc::IsStop(*stops, body_then_eos.back()));
  const std::vector<std::int32_t> length{10, 11, 12};
  EXPECT_FALSE(lc::IsStop(*stops, length.back()));
}
}  // namespace
