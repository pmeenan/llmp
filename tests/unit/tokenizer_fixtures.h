// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Shared by the JSON, tokenizer and chat tests: the fixtures under
// tests/unit/data (copied into the build tree, LLMP_TEST_DATA), the model
// files on a Spark (LLMP_TEST_MODELS, or ~/.local/share/llmp), reading
// fixture JSON, and conversations read from a chat fixture. The fixtures
// are this repository's own files: one that is missing or malformed ends
// the test binary with its name, rather than failing test by test.

#ifndef LLMP_TESTS_UNIT_TOKENIZER_FIXTURES_H_
#define LLMP_TESTS_UNIT_TOKENIZER_FIXTURES_H_

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <optional>
#include <print>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/check.h"
#include "base/json.h"
#include "chat/chat.h"

namespace llmp::test_support {

[[noreturn]] inline void BadFixture(std::string_view what) {
  std::println(stderr, "test fixture: {}", what);
  base::Fatal("a test fixture is missing or malformed");
}

inline std::string DataDir() {
  const char* dir = std::getenv("LLMP_TEST_DATA");  // NOLINT(concurrency-mt-unsafe)
  return dir != nullptr ? dir : "";
}

inline std::string ModelsDir() {
  // NOLINTNEXTLINE(concurrency-mt-unsafe): tests read, never set, the environment
  if (const char* dir = std::getenv("LLMP_TEST_MODELS"); dir != nullptr) {
    return dir;
  }
  const char* home = std::getenv("HOME");  // NOLINT(concurrency-mt-unsafe)
  return home != nullptr ? std::string(home) + "/.local/share/llmp" : "";
}

inline std::optional<std::string> ReadFile(const std::string& path,
                                           std::size_t max = std::size_t{256} << 20U) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return std::nullopt;
  }
  std::string bytes;
  bytes.resize(max);
  in.read(bytes.data(), static_cast<std::streamsize>(max));
  bytes.resize(static_cast<std::size_t>(in.gcount()));
  return bytes;
}

// A fixture file under DataDir(), parsed.
inline base::json::Document LoadJson(const std::string& relative) {
  const std::string path = DataDir() + "/" + relative;
  auto bytes = ReadFile(path);
  if (!bytes) {
    BadFixture(path);
  }
  auto doc = base::json::Parse(*bytes);
  if (!doc) {
    BadFixture(path + ": " + doc.error().ToString());
  }
  return std::move(*doc);
}

inline base::json::Value Get(base::json::Value v, std::string_view name) {
  if (const auto m = v.find(name)) {
    return *m;
  }
  BadFixture(name);
}

inline std::int64_t Int(base::json::Value v) {
  if (const auto x = v.int64()) {
    return *x;
  }
  BadFixture("an integer");
}

inline std::vector<std::int32_t> Ids(base::json::Value array) {
  std::vector<std::int32_t> ids;
  ids.reserve(array.size());
  for (std::size_t i = 0; i < array.size(); ++i) {
    ids.push_back(static_cast<std::int32_t>(Int(array.at(i))));
  }
  return ids;
}

inline chat::Role RoleFrom(std::string_view role) {
  if (role == "system") {
    return chat::Role::kSystem;
  }
  if (role == "user") {
    return chat::Role::kUser;
  }
  if (role == "assistant") {
    return chat::Role::kAssistant;
  }
  if (role == "tool") {
    return chat::Role::kTool;
  }
  BadFixture(role);
}

// A conversation from a chat fixture case; its JSON values point into the
// case's document. A tool call's arguments given as a string (the client's
// text, which a template may parse with `from_json`) are parsed into a
// document `parsed` keeps, as the chat route parses a client's.
inline chat::Conversation ConversationFrom(base::json::Value c,
                                           std::deque<base::json::Document>* parsed = nullptr) {
  using base::json::Value;
  chat::Conversation conv;
  auto text = [](std::optional<Value> v) -> std::optional<std::string> {
    if (!v || v->is_null()) {
      return std::nullopt;
    }
    return std::string(v->string());
  };
  const Value messages = Get(c, "messages");
  for (std::size_t i = 0; i < messages.size(); ++i) {
    const Value m = messages.at(i);
    chat::Message msg;
    msg.role = RoleFrom(Get(m, "role").string());
    msg.content = text(m.find("content"));
    msg.reasoning_content = text(m.find("reasoning_content"));
    if (const auto calls = m.find("tool_calls")) {
      for (std::size_t k = 0; k < calls->size(); ++k) {
        const Value call = calls->at(k);
        Value arguments = Get(call, "arguments");
        if (arguments.is_string()) {
          if (parsed == nullptr) {
            BadFixture("string arguments without a document to keep");
          }
          auto doc = base::json::Parse(arguments.string());
          if (!doc) {
            BadFixture("string arguments: " + doc.error().ToString());
          }
          arguments = parsed->emplace_back(std::move(*doc)).root();
        }
        msg.tool_calls.push_back({std::string(Get(call, "name").string()), arguments});
      }
    }
    conv.messages.push_back(std::move(msg));
  }
  if (const auto tools = c.find("tools")) {
    for (std::size_t i = 0; i < tools->size(); ++i) {
      conv.tools.push_back(tools->at(i));
    }
  }
  if (const auto gen = c.find("add_generation_prompt")) {
    conv.add_generation_prompt = gen->boolean();
  }
  if (const auto options = c.find("options")) {
    if (const auto v = options->find("enable_thinking")) {
      conv.enable_thinking = v->boolean();
    }
    if (const auto v = options->find("reasoning_effort")) {
      conv.reasoning_effort = std::string(v->string());
    }
    if (const auto v = options->find("preserve_thinking")) {
      conv.preserve_thinking = v->boolean();
    }
  }
  return conv;
}

}  // namespace llmp::test_support

#endif  // LLMP_TESTS_UNIT_TOKENIZER_FIXTURES_H_
