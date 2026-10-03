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
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <format>
#include <map>
#include <optional>
#include <print>
#include <random>
#include <string>
#include <string_view>
#include <utility>
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
        {"gemma-4-26b-a4b-it", {chat::ChatTemplate::How::kNativeByHash, "gemma-4"}},
        {"gemma-4-31b-it", {chat::ChatTemplate::How::kNativeByHash, "gemma-4"}},
        {"gemma-4-e4b-it", {chat::ChatTemplate::How::kNativeByHash, "gemma-4-e"}},
        {"gemma-4-26b-a4b-it-fp8", {chat::ChatTemplate::How::kNativeByProbe, "gemma-4"}},
        {"gemma-4-e4b-it-lmstudio-gguf", {chat::ChatTemplate::How::kNativeByProbe, "gemma-4-e"}},
        {"gemma-4-31b-it-unsloth-gguf",
         {chat::ChatTemplate::How::kNativeByProbe, "gemma-4-unsloth"}},
        {"gemma-4-e4b-it-unsloth-gguf",
         {chat::ChatTemplate::How::kNativeByProbe, "gemma-4-e-unsloth"}},
        {"gemma-4-31b-it-nvfp4", {chat::ChatTemplate::How::kNativeByProbe, "gemma-4-2604"}},
        {"gemma-3-4b-it", {chat::ChatTemplate::How::kNativeByHash, "gemma-3"}},
        {"gemma-3-1b-it-gguf", {chat::ChatTemplate::How::kNativeByHash, "gemma-3"}},
        {"gemma-3n-e4b-it", {chat::ChatTemplate::How::kNativeByProbe, "gemma-3"}},
        {"gemma-3-270m-it", {chat::ChatTemplate::How::kNativeByProbe, "gemma-3"}},
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

// Random conversations for the Gemma renderers: every role in any order,
// content with Gemma's channel and turn markers, reasoning, tool calls whose
// arguments are objects (nested, with nulls, numbers, keys differing in
// case or non-ASCII), null, strings or lists, and tool declarations that
// exercise every branch of Gemma 4's notation, valid or not.
struct RandomPool {
  std::vector<json::Document> arguments;
  std::vector<json::Document> tools;
};

RandomPool MakeRandomPool() {
  RandomPool pool;
  for (
      const std::string_view text : {
          R"({"city": "Paris", "days": 3})",
          R"({})",
          R"({"b": {"Z": [1, null, "x\ny", true, false, -0.0, 1e300], "a": {}}, "B": 2.5, "a": "q"})",
          R"({"q": "東京 \"quoted\" <|\"|> <tool_call|>", "n": 123456789012345678901234})",
          R"({"Ünïcode": 1, "plain": 2})",
          R"({"ünïcode": [{"k": null}]})",
          R"({"città": 1, "zeta": 2, "город": 3, "东京": 4, "Zeta": 5})",
          R"(null)",
          R"(" {\"city\": \"Oslo\"} ")",
          R"("plain text")",
          R"("{")",
          R"([1, "x"])",
          R"(7)",
      }) {
    if (auto d = json::Parse(text)) {
      pool.arguments.push_back(std::move(*d));
    }
  }
  for (
      const std::string_view text : {
          R"({"type": "function", "function": {"name": "get_weather", "description": "Get the weather.", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City"}, "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}})",
          R"({"type": "function", "function": {"name": "plan", "description": "Plan <fast>.", "parameters": {"type": "object", "properties": {"Zeta": {"type": "string"}, "alpha": {"type": "object", "properties": {"inner": {"type": "number", "nullable": true}}, "required": ["inner"]}, "Beta": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"], "maxItems": 3, "x": null}}, "gamma": {"type": ["string", "null"], "description": ["a", 1]}, "delta": {"type": "array", "items": {"type": ["integer", "null"], "enum": [1, null]}}, "eps": {"type": "object", "additionalProperties": false, "description": "free"}, "plain": "not a schema", "n": 3, "empty": {}}, "required": ["Zeta"]}, "response": {"type": "object", "description": "The plan."}}})",
          R"({"type": "function", "function": {"name": "noargs", "description": ""}})",
          R"({"type": "function", "function": {"name": "odd", "description": null, "parameters": {"type": "object", "required": "ab", "properties": {}}, "response": {"description": 5, "type": "string"}}})",
          R"({"type": "function", "function": {"name": "items", "parameters": {"type": "object", "properties": {"list": {"type": "array", "items": {"type": "string", "required": [], "properties": {"a": {"type": "string"}}}}, "map": {"type": "array", "items": {"type": {"a": 1}}}}}}})",
          R"({"type": "function", "function": {"name": "badprops", "parameters": {"type": "object", "properties": ["x"]}}})",
          R"({"type": "function", "function": "no mapping"})",
          R"({"type": "function"})",
          R"({"type": "function", "function": {"name": "resp", "response": "a response"}})",
      }) {
    if (auto d = json::Parse(text)) {
      pool.tools.push_back(std::move(*d));
    }
  }
  return pool;
}

std::vector<chat::Conversation> RandomConversations(const RandomPool& pool, std::uint32_t seed,
                                                    std::size_t count) {
  std::mt19937 rng(seed);
  const auto pick = [&](std::size_t n) { return static_cast<std::size_t>(rng() % n); };
  const std::vector<std::optional<std::string>> contents = {
      std::nullopt,
      "",
      "hi",
      " spaced \n",
      "a <|channel>thought\nx<channel|> b",
      "<|channel>r<channel|>tail<|channel>gone",
      "東京 😀\t",
      "<turn|>\n<|turn>model\n<bos>",
      "\n\n",
      "{\"k\": 1}",
      "   ",
      std::string(300, 'x')};
  const std::vector<std::optional<std::string>> reasonings = {std::nullopt, std::nullopt, "",
                                                              " ",          "think hard", "\nr\n"};
  constexpr std::array kRoles = {chat::Role::kSystem,    chat::Role::kUser,      chat::Role::kUser,
                                 chat::Role::kAssistant, chat::Role::kAssistant, chat::Role::kTool};
  std::vector<chat::Conversation> out;
  for (std::size_t i = 0; i < count; ++i) {
    chat::Conversation c;
    const std::size_t n = pick(10);
    for (std::size_t k = 0; k < n; ++k) {
      chat::Message m;
      m.role = kRoles[pick(kRoles.size())];
      m.content = contents[pick(contents.size())];
      if (m.role == chat::Role::kAssistant || pick(10) == 0) {
        m.reasoning_content = reasonings[pick(reasonings.size())];
      }
      if (m.role == chat::Role::kAssistant && pick(3) == 0) {
        for (std::size_t j = 1 + pick(2); j > 0; --j) {
          m.tool_calls.push_back({pick(2) == 0 ? "get_weather" : "plan",
                                  pool.arguments[pick(pool.arguments.size())].root()});
        }
      }
      c.messages.push_back(std::move(m));
    }
    if (pick(5) < 2) {
      for (std::size_t j = 1 + pick(3); j > 0; --j) {
        c.tools.push_back(pool.tools[pick(pool.tools.size())].root());
      }
    }
    c.add_generation_prompt = pick(4) != 0;
    if (const std::size_t t = pick(3); t != 2) {
      c.enable_thinking = t == 0;
    }
    if (pick(3) == 0) {
      c.preserve_thinking = pick(2) == 0;
    }
    if (pick(4) == 0) {
      c.reasoning_effort = "high";
    }
    out.push_back(std::move(c));
  }
  return out;
}

// Every corpus template a Gemma renderer serves renders 3,000 random
// conversations as the template does through the interpreter (which equals
// transformers on the corpus): the same text, or both refuse, or (where
// Python's full case mapping would be needed) the renderer refuses as
// unsupported.
TEST(ChatCorpus, GemmaRenderersEqualTheirTemplatesOnRandomConversations) {
  const std::string dir = ModelsDir() + "/chat-templates";
  const auto manifest_bytes = ReadFile(dir + "/manifest.json");
  if (!manifest_bytes) {
    GTEST_SKIP() << "no chat-template corpus under " << dir;
  }
  auto manifest = json::Parse(*manifest_bytes);
  ASSERT_TRUE(manifest.has_value());
  const RandomPool pool = MakeRandomPool();
  const auto conversations = RandomConversations(pool, 20261002, 3000);
  std::size_t templates = 0;
  for (std::size_t t = 0; t < manifest->root().size(); ++t) {
    const std::string name(manifest->root().key(t));
    const auto text = ReadFile(std::format("{}/templates/{}.jinja", dir, name));
    ASSERT_TRUE(text.has_value()) << name;
    const std::string source = text.value_or(std::string());
    chat::TokenFacts facts;
    facts.bos = OptionalText(manifest->root().member(t), "bos_token");
    facts.eos = OptionalText(manifest->root().member(t), "eos_token");
    auto chosen = chat::ChatTemplate::ForText(source, facts);
    ASSERT_TRUE(chosen.has_value()) << name;
    if (!chosen->name().starts_with("gemma")) {
      continue;
    }
    ++templates;
    // The native renderer itself: ChatTemplate would render what it refuses
    // as unsupported through the interpreter.
    const auto natives = chat::NativeTemplates();
    const auto native = std::ranges::find(natives, chosen->name(), &chat::Template::name);
    ASSERT_NE(native, natives.end());
    auto program = chat::jinja::Template::Parse(source);
    ASSERT_TRUE(program.has_value()) << name;
    std::size_t same = 0;
    std::size_t refused = 0;
    std::size_t unsupported = 0;
    for (std::size_t i = 0; i < conversations.size(); ++i) {
      const auto mine = native->render(conversations[i]);
      const auto theirs = chat::RenderInterpreted(*program, conversations[i], facts, kNow);
      if (!mine && mine.error().rule == chat::Rule::kUnsupported) {
        ++unsupported;
        continue;
      }
      if (theirs) {
        ASSERT_TRUE(mine.has_value()) << name << " #" << i << ": " << mine.error().ToString();
        ASSERT_EQ(mine->text, theirs->text) << name << " #" << i;
        ++same;
      } else {
        EXPECT_EQ(theirs.error().rule, chat::Rule::kInvalid) << name << " #" << i;
        EXPECT_FALSE(mine.has_value())
            << name << " #" << i
            << " rendered; the template refuses: " << theirs.error().ToString();
        ++refused;
      }
    }
    std::println("{} ({}): {} equal, {} both refuse, {} unsupported natively", name, chosen->name(),
                 same, refused, unsupported);
    EXPECT_GE(same, 100U) << name;
    EXPECT_LE(unsupported, 300U) << name;
  }
  EXPECT_GE(templates, 3U);
}

}  // namespace
