// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// CPU-only preparation: METADATA TEXT NEW_OUT literal|chat. No model execution.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <optional>
#include <print>
#include <span>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "base/json.h"
#include "base/sha256.h"
#include "chat/chat.h"
#include "tokenizer/gguf.h"
#include "tokenizer/tokenizer.h"

namespace {
namespace fs = std::filesystem;
namespace tok = llmp::tokenizer;
using Result = std::expected<void, std::string>;
std::string Quoted(std::string_view text) {
  std::string out;
  llmp::base::json::AppendQuoted(text, out);
  return out;
}
std::string Digest(std::string_view text) {
  return llmp::base::ToHex(llmp::base::Sha256{}.Update(text).Finish());
}
std::expected<std::string, std::string> Read(const fs::path& path, std::uintmax_t cap) {
  std::error_code error;
  if (!fs::is_regular_file(path, error) || error) return std::unexpected("not a regular file");
  const auto size = fs::file_size(path, error);
  if (error || size > cap) return std::unexpected("input exceeds byte bound");
  std::string bytes(static_cast<std::size_t>(size), '\0');
  std::ifstream file(path, std::ios::binary);
  if (!file || !file.read(bytes.data(), static_cast<std::streamsize>(bytes.size())) ||
      file.peek() != std::char_traits<char>::eof())
    return std::unexpected("input changed or could not be read completely");
  return bytes;
}
Result Write(const fs::path& path, std::string_view bytes) {
  std::ofstream file(path, std::ios::binary);
  file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  file.close();
  if (!file) return std::unexpected("output write or close failed");
  return {};
}
Result Prepare(int argc, char** argv) {
  if (argc != 5) return std::unexpected("usage: METADATA TEXT NEW_OUT literal|chat");
  const std::string mode(argv[4]);
  if (mode != "literal" && mode != "chat") return std::unexpected("unknown mode");
  auto metadata = Read(argv[1], 32U << 20U);
  auto text = Read(argv[2], 4U << 20U);
  if (!metadata) return std::unexpected(metadata.error());
  if (!text) return std::unexpected(text.error());
  auto parsed = tok::ReadGgufTokenizer(std::as_bytes(std::span(*metadata)));
  if (!parsed) return std::unexpected(parsed.error().ToString());
  if (parsed->spec.pre_tokenizer != tok::PreTokenizer::kGemma4 ||
      parsed->spec.tokens.size() != 262144 || parsed->spec.bos != 2 || !parsed->spec.add_bos ||
      parsed->spec.add_eos || !parsed->has_chat_template)
    return std::unexpected("expected Gemma4 vocabulary/BOS/template contract");
  const auto template_sha = Digest(parsed->chat_template);
  auto tokenizer = tok::Tokenizer::Create(std::move(parsed->spec));
  if (!tokenizer) return std::unexpected(tokenizer.error().ToString());
  std::vector<tok::TokenId> ids;
  std::string rendered;
  std::string renderer = "none";
  std::string rendering_how = "none";
  if (mode == "literal") {
    if (auto encoded =
            tokenizer->Encode(*text, {.add_bos_eos = true, .max_tokens = 1U << 20U}, ids);
        !encoded)
      return std::unexpected(encoded.error().ToString());
  } else {
    llmp::chat::jinja::Limits limits;
    limits.max_string_bytes = 4U << 20U;
    limits.max_output_bytes = 4U << 20U;
    auto model_template = llmp::chat::ChatTemplate::ForText(
        parsed->chat_template, llmp::chat::TokenFacts::From(*tokenizer), limits);
    if (!model_template) return std::unexpected(model_template.error());
    llmp::chat::Conversation conversation;
    conversation.enable_thinking = false;
    conversation.max_render_bytes = limits.max_output_bytes;
    conversation.messages.push_back({.role = llmp::chat::Role::kUser,
                                     .content = *text,
                                     .reasoning_content = std::nullopt,
                                     .tool_calls = {}});
    renderer = model_template->name();
    switch (model_template->how()) {
      case llmp::chat::ChatTemplate::How::kNativeByHash:
        rendering_how = "native-by-hash";
        break;
      case llmp::chat::ChatTemplate::How::kNativeByProbe:
        rendering_how = "native-by-probe";
        break;
      case llmp::chat::ChatTemplate::How::kInterpreted:
        rendering_how = "interpreted";
        break;
    }
    auto result = model_template->Render(conversation);
    if (!result) return std::unexpected(result.error().ToString());
    rendered = std::move(result->text);
    if (auto encoded =
            tokenizer->EncodeMarked(rendered, result->specials, {.max_tokens = 1U << 20U}, ids);
        !encoded)
      return std::unexpected(encoded.error().ToString());
  }
  if (ids.empty() || ids.front() != 2 || std::count(ids.begin(), ids.end(), 2) != 1 ||
      std::ranges::any_of(ids, [](auto id) { return id < 0 || id >= 262144; }))
    return std::unexpected("IDs require exactly one leading BOS and the fixed vocabulary");
  std::string raw;
  raw.reserve(ids.size() * 4);
  for (auto id : ids) {
    const auto bits = static_cast<std::uint32_t>(id);
    for (unsigned shift = 0; shift < 32; shift += 8)
      raw.push_back(static_cast<char>((bits >> shift) & 255U));
  }
  const fs::path out(argv[3]);
  std::error_code error;
  if (!fs::create_directory(out, error) || error)
    return std::unexpected("output directory must be fresh with an existing parent");
  fs::permissions(out, fs::perms::owner_all, fs::perm_options::replace, error);
  if (error) return std::unexpected("private output directory permissions failed");
  if (auto wrote = Write(out / "ids.i32", raw); !wrote) return wrote;
  if (mode == "chat") {
    if (auto wrote = Write(out / "rendered.txt", rendered); !wrote) return wrote;
    if (auto wrote = Write(out / "template.jinja", parsed->chat_template); !wrote) return wrote;
  }
  const auto receipt = std::format(
      "{{\"schema\":1,\"mode\":\"{}\",\"model_loaded\":false,\"metadata_sha256\":\"{}\","
      "\"text_sha256\":\"{}\",\"template_sha256\":\"{}\",\"rendered_sha256\":\"{}\","
      "\"input_sha256\":\"{}\",\"input_bytes\":{},\"tokens\":{},\"bos_count\":1,"
      "\"vocab\":262144,\"enable_thinking\":false,\"add_generation_prompt\":true,"
      "\"renderer\":{},\"rendering_how\":{}}}\n",
      mode, Digest(*metadata), Digest(*text), template_sha, Digest(rendered), Digest(raw),
      raw.size(), ids.size(), Quoted(renderer), Quoted(rendering_how));
  if (auto wrote = Write(out / "receipt.json", receipt); !wrote) return wrote;
  std::println("GEMMA_TOKENIZE mode={} tokens={} input_sha256={}", mode, ids.size(), Digest(raw));
  return {};
}
}  // namespace
int main(int argc, char** argv) {
  auto result = Prepare(argc, argv);
  if (!result) {
    std::println(stderr, "{}", result.error());
    return 1;
  }
  return 0;
}
