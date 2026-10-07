// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The approved artifact supplies the template; no copied checkpoint template
// or new family renderer is needed for its system-role refusal.
#include <gtest/gtest.h>
#include <unistd.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "base/sha256.h"
#include "chat/chat.h"
#include "expected_error.h"
#include "model/gemma2.h"
#include "runtime/model_settings.h"
#include "tokenizer/tokenizer.h"

namespace {
namespace chat = jitllm::chat;
namespace rt = jitllm::runtime;
namespace tok = jitllm::tokenizer;
class Gemma2ChatAssets : public ::testing::Test {
 protected:
  void SetUp() override {
    auto artifact = jitllm::artifact::Artifact::Open(
        "/home/pmeenan/.local/share/jitllm/gemma2-import-20261007/artifacts/"
        "eb18d30d0a7de3a95c7b6994b65a12a057ffbf42866add6f128873de8b7aa870",
        {.expected_id = {}, .trusted_owner = ::geteuid()});
    ASSERT_TRUE(artifact) << jitllm::test_support::Failed(artifact)->ToString();
    auto bound = jitllm::model::BindApprovedGemma2(*artifact);
    ASSERT_TRUE(bound) << *jitllm::test_support::Failed(bound);
    auto facts = rt::ArtifactFactsOf(*artifact, nullptr);
    ASSERT_TRUE(facts) << *jitllm::test_support::Failed(facts);
    EXPECT_EQ(facts->architecture, "gemma2");
    EXPECT_EQ(facts->trained_context, 8192);
    auto settings = rt::ResolveSettings({}, *facts, nullptr, false);
    ASSERT_TRUE(settings) << *jitllm::test_support::Failed(settings);
    EXPECT_EQ(settings->context.value, 8192U);
    EXPECT_EQ(settings->prefill_chunk.value, 128U);
    EXPECT_EQ(settings->max_slots.value, 1U);
    auto assets = rt::ReadChatAssets(*artifact, {}, ::geteuid());
    ASSERT_TRUE(assets) << *jitllm::test_support::Failed(assets);
    ASSERT_TRUE(assets->chat_template);
    ASSERT_EQ(assets->chat_template->size(), 591U);
    EXPECT_EQ(jitllm::base::ToHex(jitllm::base::Sha256{}.Update(*assets->chat_template).Finish()),
              "ecd6ae513fe103f0eb62e8ab5bfa8d0fe45c1074fa398b089c93a7e70c15cfd6");
    auto created = tok::Tokenizer::Create(std::move(assets->spec));
    ASSERT_TRUE(created) << jitllm::test_support::Failed(created)->ToString();
    tokenizer = std::make_unique<tok::Tokenizer>(std::move(*created));
    auto chosen =
        chat::ChatTemplate::ForText(*assets->chat_template, chat::TokenFacts::From(*tokenizer));
    ASSERT_TRUE(chosen) << *jitllm::test_support::Failed(chosen);
    EXPECT_EQ(chosen->how(), chat::ChatTemplate::How::kInterpreted);
    renderer = std::make_unique<chat::ChatTemplate>(std::move(*chosen));
  }
  std::unique_ptr<tok::Tokenizer> tokenizer;
  std::unique_ptr<chat::ChatTemplate> renderer;
};

TEST_F(Gemma2ChatAssets, ActualTemplateRefusesSystemAndBrokenAlternation) {
  chat::Conversation c;
  c.messages = {{chat::Role::kSystem, "rules", {}, {}}, {chat::Role::kUser, "hi", {}, {}}};
  auto rendered = renderer->Render(c);
  ASSERT_FALSE(rendered);
  EXPECT_EQ(rendered.error().rule, chat::Rule::kInvalid);
  c.messages.erase(c.messages.begin());
  c.messages.push_back({chat::Role::kUser, "another user", {}, {}});
  rendered = renderer->Render(c);
  ASSERT_FALSE(rendered);
  EXPECT_EQ(rendered.error().rule, chat::Rule::kInvalid);
  c.messages = {{chat::Role::kAssistant, "assistant first", {}, {}}};
  rendered = renderer->Render(c);
  ASSERT_FALSE(rendered);
  EXPECT_EQ(rendered.error().rule, chat::Rule::kInvalid);
}

TEST_F(Gemma2ChatAssets, AlternatingTrimmedTurnsAndGenerationPrefixMatchTheActualTemplate) {
  chat::Conversation c;
  c.messages = {{chat::Role::kUser, " \n hello <end_of_turn> \t", {}, {}},
                {chat::Role::kAssistant, " answer \n", {}, {}},
                {chat::Role::kUser, " next ", {}, {}}};
  const std::string expected =
      "<bos><start_of_turn>user\nhello <end_of_turn><end_of_turn>\n"
      "<start_of_turn>model\nanswer<end_of_turn>\n"
      "<start_of_turn>user\nnext<end_of_turn>\n";
  auto rendered = renderer->Render(c);
  ASSERT_TRUE(rendered) << jitllm::test_support::Failed(rendered)->ToString();
  EXPECT_EQ(rendered->text, expected + "<start_of_turn>model\n");
  std::vector<std::string> marked;
  const auto user_control = rendered->text.find("hello <end_of_turn>") + 6;
  for (const auto& special : rendered->specials) {
    marked.emplace_back(rendered->text.substr(special.offset, special.length));
    EXPECT_NE(special.offset, user_control);  // Content cannot inject a stop control.
  }
  EXPECT_EQ(marked, (std::vector<std::string>{"<bos>", "<start_of_turn>", "<end_of_turn>",
                                              "<start_of_turn>", "<end_of_turn>", "<start_of_turn>",
                                              "<end_of_turn>", "<start_of_turn>"}));
  c.add_generation_prompt = false;
  rendered = renderer->Render(c);
  ASSERT_TRUE(rendered) << jitllm::test_support::Failed(rendered)->ToString();
  EXPECT_EQ(rendered->text, expected);
}

TEST_F(Gemma2ChatAssets, ActualEndOfTurnAndEosRemainDistinctStops) {
  EXPECT_EQ(tokenizer->bos(), 2);
  EXPECT_EQ(tokenizer->eos(), 1);
  ASSERT_EQ(renderer->stop(), std::vector<std::string>{"<end_of_turn>"});
  const auto stops = chat::StopTokens(renderer->stop(), *tokenizer);
  ASSERT_TRUE(stops) << jitllm::test_support::Failed(stops)->ToString();
  EXPECT_EQ(*stops, std::vector<tok::TokenId>{107});
  EXPECT_NE(stops->front(), *tokenizer->eos());
}
}  // namespace
