// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Real chat templates against transformers' renderings
// (docs/experiments/chat-template-corpus): every template of the corpus,
// through the interpreter and through what ChatTemplate chooses for it, on
// every conversation of tests/unit/data/chat/corpus-conversations.json. The
// templates and references live outside the repository (their licenses
// vary), under the models directory's chat-templates/; the test skips where
// they are absent and fails where a template's bytes differ from its
// manifest's hash.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <deque>
#include <format>
#include <map>
#include <optional>
#include <print>
#include <string>
#include <string_view>
#include <vector>

#include "base/json.h"
#include "base/sha256.h"
#include "chat/chat.h"
#include "chat/jinja.h"
#include "tokenizer_fixtures.h"

namespace {

namespace chat = jitllm::chat;
namespace json = jitllm::base::json;
using jitllm::test_support::ConversationFrom;
using jitllm::test_support::Get;
using jitllm::test_support::LoadJson;
using jitllm::test_support::ModelsDir;
using jitllm::test_support::ReadFile;

const jitllm::chat::jinja::CivilTime kNow{.year = 2026,
                                          .month = 10,
                                          .day = 2,
                                          .hour = 12,
                                          .minute = 34,
                                          .second = 56,
                                          .weekday = 5,
                                          .yearday = 274};

std::optional<std::string> OptionalText(json::Value v, std::string_view key) {
  const auto m = v.find(key);
  if (!m || m->is_null()) {
    return std::nullopt;
  }
  return std::string(m->string());
}

// How each corpus template must be chosen; the rest are interpreted.
const std::map<std::string, std::pair<chat::ChatTemplate::How, std::string>, std::less<>>
    kExpectedNative = {
        {"deepseek-v4-0731-gguf",
         {chat::ChatTemplate::How::kNativeByHash, "deepseek-v4-flash-0731"}},
        {"deepseek-v4-chat-v2-gguf",
         {chat::ChatTemplate::How::kNativeByHash, "deepseek-v4-flash-chat-v2"}},
        {"qwen3.8-flash-next-nvfp4",
         {chat::ChatTemplate::How::kNativeByHash, "qwen3.8-flash-next"}},
        {"qwen3.8-flash-next", {chat::ChatTemplate::How::kNativeByHash, "qwen3.8-flash-next"}},
        {"qwen3.8-27b", {chat::ChatTemplate::How::kNativeByHash, "qwen3.8-flash-next"}},
        {"qwen3.8-flash-next-unsloth-gguf",
         {chat::ChatTemplate::How::kNativeByProbe, "qwen3.8-flash-next-unsloth"}},
        {"qwen3.8-27b-unsloth-gguf",
         {chat::ChatTemplate::How::kNativeByProbe, "qwen3.8-flash-next-unsloth"}},
};

TEST(ChatCorpus, EveryTemplateRendersAsTransformersDoes) {
  const std::string dir = ModelsDir() + "/chat-templates";
  const auto manifest_bytes = ReadFile(dir + "/manifest.json");
  const auto reference_bytes = ReadFile(dir + "/references.json");
  if (!manifest_bytes || !reference_bytes) {
    GTEST_SKIP() << "no chat-template corpus under " << dir;
  }
  auto manifest = json::Parse(*manifest_bytes);
  auto references = json::Parse(*reference_bytes);
  ASSERT_TRUE(manifest.has_value() && references.has_value());
  const json::Document convs = LoadJson("chat/corpus-conversations.json");
  const json::Value conversations = Get(convs.root(), "conversations");
  const json::Value templates = Get(references->root(), "templates");
  std::size_t compared = 0;
  std::size_t interpreted_templates = 0;
  for (std::size_t t = 0; t < manifest->root().size(); ++t) {
    const std::string name(manifest->root().key(t));
    const json::Value entry = manifest->root().member(t);
    const auto text = ReadFile(std::format("{}/templates/{}.jinja", dir, name));
    ASSERT_TRUE(text.has_value()) << name;
    const std::string source = text.value_or(std::string());
    ASSERT_EQ(jitllm::base::ToHex(jitllm::base::Sha256().Update(source).Finish()),
              Get(entry, "sha256").string())
        << name;
    chat::TokenFacts facts;
    facts.bos = OptionalText(entry, "bos_token");
    facts.eos = OptionalText(entry, "eos_token");
    auto program = chat::jinja::Template::Parse(source);
    ASSERT_TRUE(program.has_value()) << name << ": " << program.error().ToString();
    auto chosen = chat::ChatTemplate::ForText(source, facts);
    ASSERT_TRUE(chosen.has_value()) << name << ": " << chosen.error();
    if (const auto it = kExpectedNative.find(name); it != kExpectedNative.end()) {
      EXPECT_EQ(chosen->how(), it->second.first) << name;
      EXPECT_EQ(chosen->name(), it->second.second) << name;
    } else {
      EXPECT_EQ(chosen->how(), chat::ChatTemplate::How::kInterpreted) << name;
      ++interpreted_templates;
    }
    const json::Value cases = Get(Get(templates, name), "cases");
    for (std::size_t i = 0; i < conversations.size(); ++i) {
      const json::Value conv = conversations.at(i);
      const std::string_view case_name = Get(conv, "name").string();
      const json::Value expected = Get(cases, case_name);
      std::deque<json::Document> arguments;
      const chat::Conversation c = ConversationFrom(conv, &arguments);
      // The interpreter alone, whatever ChatTemplate chose.
      const auto mine = chat::RenderInterpreted(*program, c, facts, kNow);
      if (const auto want = expected.find("text")) {
        ASSERT_TRUE(mine.has_value())
            << name << "/" << case_name << ": " << mine.error().ToString();
        EXPECT_EQ(mine->text, want->string()) << name << "/" << case_name;
        ++compared;
      } else {
        EXPECT_FALSE(mine.has_value())
            << name << "/" << case_name
            << " rendered; transformers: " << Get(expected, "error").string();
        if (!mine) {
          EXPECT_EQ(mine.error().rule, chat::Rule::kInvalid)
              << name << "/" << case_name << ": " << mine.error().ToString();
        }
      }
      // What serving uses: the same text, or a native refusal of a case it
      // does not implement.
      const auto served = chosen->Render(c, kNow);
      if (const auto want = expected.find("text")) {
        if (served) {
          EXPECT_EQ(served->text, want->string()) << name << "/" << case_name << " (served)";
        } else {
          EXPECT_EQ(served.error().rule, chat::Rule::kUnsupported)
              << name << "/" << case_name << ": " << served.error().ToString();
        }
      } else {
        EXPECT_FALSE(served.has_value()) << name << "/" << case_name << " (served)";
      }
    }
  }
  std::println("compared {} renderings; {} templates interpreted", compared, interpreted_templates);
  EXPECT_GE(compared, 400U);
  EXPECT_GE(interpreted_templates, 15U);
}

// A long conversation (about 1 MiB in 3,000 messages, as a long context
// holds) renders through every template within the interpreter's bounds.
TEST(ChatCorpus, LongConversationsStayWithinBounds) {
  const std::string dir = ModelsDir() + "/chat-templates/templates";
  const auto manifest_bytes = ReadFile(ModelsDir() + "/chat-templates/manifest.json");
  if (!manifest_bytes) {
    GTEST_SKIP() << "no chat-template corpus";
  }
  auto manifest = json::Parse(*manifest_bytes);
  ASSERT_TRUE(manifest.has_value());
  chat::Conversation c;
  c.messages.push_back({chat::Role::kSystem, "You are helpful.", std::nullopt, {}});
  const std::string filler(340, 'x');
  for (int i = 0; i < 1500; ++i) {
    c.messages.push_back(
        {chat::Role::kUser, "question " + std::to_string(i) + " " + filler, std::nullopt, {}});
    c.messages.push_back(
        {chat::Role::kAssistant, "answer " + std::to_string(i) + " " + filler, std::nullopt, {}});
  }
  c.messages.push_back({chat::Role::kUser, "last", std::nullopt, {}});
  double slowest = 0;
  for (std::size_t t = 0; t < manifest->root().size(); ++t) {
    const std::string name(manifest->root().key(t));
    const auto text = ReadFile(std::format("{}/{}.jinja", dir, name));
    ASSERT_TRUE(text.has_value());
    const std::string source = text.value_or(std::string());
    auto program = chat::jinja::Template::Parse(source);
    ASSERT_TRUE(program.has_value());
    chat::TokenFacts facts;
    facts.bos = OptionalText(manifest->root().member(t), "bos_token");
    const auto started = std::chrono::steady_clock::now();
    const auto r = chat::RenderInterpreted(*program, c, facts, kNow);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    slowest = std::max(slowest, seconds);
    // A template may reject the conversation (Gemma's alternation rules), but
    // never for a bound.
    if (!r) {
      EXPECT_EQ(r.error().rule, chat::Rule::kInvalid) << name << ": " << r.error().ToString();
    } else {
      EXPECT_GT(r->text.size(), std::size_t{1} << 20U) << name;
    }
    std::println("{}: {:.3f} s{}", name, seconds, r ? "" : " (the template refuses it)");
  }
  EXPECT_LT(slowest, 5.0);
}

// The end-of-turn token an interpreted template places after an assistant
// message, given the vocabulary's control tokens.
TEST(ChatCorpus, StopTokensComeFromTheTemplate) {
  const std::string dir = ModelsDir() + "/chat-templates/templates";
  const std::vector<std::pair<std::string, std::string>> expected = {
      {"llama-3.3-70b", "<|eot_id|>"},
      {"gemma-3-4b-it", "<end_of_turn>"},
      {"qwen2.5-7b-instruct", "<|im_end|>"},
      {"phi-4", "<|im_end|>"},
  };
  chat::TokenFacts facts;
  facts.control = {
      "<|eot_id|>",      "<|start_header_id|>", "<|end_header_id|>", "<|im_start|>", "<|im_end|>",
      "<start_of_turn>", "<end_of_turn>",       "<|begin_of_text|>", "<|im_sep|>",   "<|user|>",
      "<|assistant|>",   "<|system|>",          "<|endoftext|>",     "[gMASK]",      "<sop>"};
  for (const auto& [name, stop] : expected) {
    const auto text = ReadFile(std::format("{}/{}.jinja", dir, name));
    if (!text) {
      GTEST_SKIP() << "no chat-template corpus under " << dir;
    }
    auto chosen = chat::ChatTemplate::ForText(text.value_or(std::string()), facts);
    ASSERT_TRUE(chosen.has_value()) << name << ": " << chosen.error();
    ASSERT_EQ(chosen->stop().size(), 1U) << name;
    EXPECT_EQ(chosen->stop()[0], stop) << name;
  }
}

}  // namespace
