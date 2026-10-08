// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <optional>
#include <print>
#include <span>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "base/json.h"
#include "base/sha256.h"
#include "chat/chat.h"
#include "long_context_tasks.h"
#include "tokenizer/gguf.h"
#include "tokenizer/tokenizer.h"

namespace {
namespace lc = llmp::benchmarks::long_context;
namespace js = llmp::base::json;
namespace fs = std::filesystem;
namespace tok = llmp::tokenizer;
using Result = std::expected<void, std::string>;

std::string Digest(std::string_view bytes) {
  return llmp::base::ToHex(llmp::base::Sha256{}.Update(bytes).Finish());
}
std::string IdDigest(std::span<const std::int32_t> ids) {
  llmp::base::Sha256 sha;
  for (auto id : ids) {
    const auto bits = static_cast<std::uint32_t>(id);
    const std::array<std::byte, 4> bytes{
        static_cast<std::byte>(bits & 255U), static_cast<std::byte>((bits >> 8U) & 255U),
        static_cast<std::byte>((bits >> 16U) & 255U), static_cast<std::byte>(bits >> 24U)};
    sha.Update(bytes);
  }
  return llmp::base::ToHex(sha.Finish());
}
std::string Quoted(std::string_view text) {
  std::string out;
  js::AppendQuoted(text, out);
  return out;
}
std::expected<std::string, std::string> Read(const fs::path& path, std::size_t cap) {
  std::error_code error;
  if (!fs::is_regular_file(path, error) || error)
    return std::unexpected("not a regular input file");
  const auto bytes = fs::file_size(path, error);
  if (error || bytes > cap) return std::unexpected("input file exceeds its bound");
  std::string result(static_cast<std::size_t>(bytes), '\0');
  std::ifstream file(path, std::ios::binary);
  if (!file || !file.read(result.data(), static_cast<std::streamsize>(result.size())) ||
      file.peek() != std::char_traits<char>::eof())
    return std::unexpected("input file changed or could not be read completely");
  return result;
}
Result Write(const fs::path& path, std::string_view bytes) {
  std::ofstream file(path, std::ios::binary);
  file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  file.close();
  if (!file) return std::unexpected("output write or close failed");
  return {};
}
Result Fresh(const fs::path& out) {
  std::error_code error;
  if (!fs::create_directory(out, error) || error)
    return std::unexpected("output directory must be fresh with an existing parent");
  return {};
}

class NativeCodec final : public lc::Codec {
 public:
  NativeCodec(tok::Tokenizer tokenizer, const llmp::chat::Template* model_template,
              std::string metadata_sha, std::vector<std::int32_t> stops)
      : tokenizer_(std::move(tokenizer)),
        template_(model_template),
        metadata_sha_(std::move(metadata_sha)),
        stops_(std::move(stops)) {}
  std::expected<lc::Encoding, std::string> Encode(std::string_view user) const override {
    llmp::chat::Conversation conversation;
    conversation.enable_thinking = false;
    conversation.messages.push_back({.role = llmp::chat::Role::kUser,
                                     .content = std::string(user),
                                     .reasoning_content = std::nullopt,
                                     .tool_calls = {}});
    auto rendered = template_->render(conversation);
    if (!rendered) return std::unexpected(rendered.error().ToString());
    lc::Encoding result;
    result.rendered = std::move(rendered->text);
    if (auto encoded = tokenizer_.EncodeMarked(result.rendered, rendered->specials,
                                               {.max_tokens = 131072}, result.ids);
        !encoded)
      return std::unexpected(encoded.error().ToString());
    std::string decoded;
    result.byte_ends.reserve(result.ids.size());
    for (auto id : result.ids) {
      if (auto read = tokenizer_.Decode(std::span(&id, 1), {.control_tokens = true}, decoded);
          !read)
        return std::unexpected(read.error().ToString());
      result.byte_ends.push_back(decoded.size());
    }
    if (decoded != result.rendered)
      return std::unexpected("native IDs do not reconstruct the full rendered bytes");
    return result;
  }
  std::expected<std::string, std::string> Decode(std::span<const std::int32_t> ids) const {
    std::string response;
    if (auto decoded = tokenizer_.Decode(ids, {}, response); !decoded)
      return std::unexpected(decoded.error().ToString());
    return response;
  }
  std::string_view metadata_sha() const { return metadata_sha_; }
  std::string_view template_sha() const { return template_->sha256; }
  std::span<const std::int32_t> stops() const { return stops_; }

 private:
  tok::Tokenizer tokenizer_;
  const llmp::chat::Template* template_;
  std::string metadata_sha_;
  std::vector<std::int32_t> stops_;
};
std::expected<NativeCodec, std::string> OpenCodec(const fs::path& path) {
  auto artifact = llmp::artifact::Artifact::Open(path);
  if (!artifact) return std::unexpected(artifact.error().ToString());
  if (artifact->id() != lc::kArtifact || artifact->model().architecture != "deepseek4")
    return std::unexpected("fixed HCA-quality artifact required");
  std::optional<std::string> metadata;
  for (const auto& file : artifact->files()) {
    if (file.role == llmp::artifact::FileRole::kSourceMetadata && file.path.ends_with(".kv.gguf") &&
        file.path.contains("-00001-of-")) {
      if (metadata) return std::unexpected("ambiguous first-shard tokenizer metadata");
      metadata = file.path.substr(5);
    }
  }
  if (!metadata) return std::unexpected("missing authenticated first-shard metadata");
  auto bytes = artifact->ReadMetadata(*metadata);
  if (!bytes) return std::unexpected(bytes.error().ToString());
  auto read = tok::ReadGgufTokenizer(std::as_bytes(std::span(*bytes)));
  if (!read) return std::unexpected(read.error().ToString());
  if (!read->has_chat_template || read->spec.tokens.size() != 129280 ||
      read->spec.pre_tokenizer != tok::PreTokenizer::kDeepSeekV3)
    return std::unexpected("fixed native tokenizer/template shape differs");
  auto model_template = llmp::chat::FindTemplateForText(read->chat_template);
  if (!model_template) return std::unexpected(model_template.error());
  auto tokenizer = tok::Tokenizer::Create(std::move(read->spec));
  if (!tokenizer) return std::unexpected(tokenizer.error().ToString());
  auto stops = llmp::chat::StopTokens((*model_template)->stop, *tokenizer);
  if (!stops) return std::unexpected(stops.error().ToString());
  const auto eos = tokenizer->eos();
  if (eos && !lc::IsStop(*stops, *eos)) stops->push_back(*eos);
  if (stops->empty() || stops->size() > 16)
    return std::unexpected("missing or excessive native stops");
  return NativeCodec(std::move(*tokenizer), *model_template, Digest(*bytes), std::move(*stops));
}
std::string Stops(std::span<const std::int32_t> stops) {
  std::string result;
  for (auto stop : stops) result += (result.empty() ? "" : ",") + std::to_string(stop);
  return result;
}

Result Prepare(const NativeCodec& codec, const fs::path& out) {
  if (auto fresh = Fresh(out); !fresh) return fresh;
  // Construct the complete fixed set before publishing any successful
  // receipt. A failed bounded construction leaves an explicit failure.
  std::vector<lc::Prepared> tasks;
  for (const auto& b : lc::FixedCases()) {
    auto prepared = lc::Prepare(b, codec);
    if (!prepared) {
      const auto failure = std::format("{{\"complete\":false,\"case\":{},\"error\":{}}}\n",
                                       Quoted(b.name), Quoted(prepared.error()));
      if (auto saved = Write(out / "receipt.json", failure); !saved) return saved;
      return std::unexpected(b.name + ": " + prepared.error());
    }
    tasks.push_back(std::move(*prepared));
  }
  std::string receipt = std::format(
      "{{\"complete\":true,\"model_loaded\":false,\"protocol\":{},\"seed\":{},"
      "\"artifact\":{},\"metadata_sha256\":{},\"template_sha256\":{},"
      "\"enable_thinking\":false,\"stop_ids\":[{}],\"max_rows\":2048,"
      "\"compact_experts\":true,\"frontier_head\":true,\"wide_sparse\":true,"
      "\"exact\":false,\"q2_d2r\":false,\"mtp\":false,\"cases\":[",
      Quoted(lc::kProtocol), lc::kSeed, Quoted(lc::kArtifact), Quoted(codec.metadata_sha()),
      Quoted(codec.template_sha()), Stops(codec.stops()));
  bool comma = false;
  for (const auto& task : tasks) {
    const auto& b = task.blueprint;
    std::string ids = b.name + '\t';
    for (std::size_t i = 0; i < task.encoded.ids.size(); ++i)
      ids += (i == 0 ? "" : " ") + std::to_string(task.encoded.ids[i]);
    ids += '\n';
    if (auto saved = Write(out / (b.name + ".ids.tsv"), ids); !saved) return saved;
    if (auto saved = Write(out / (b.name + ".user.txt"), task.user); !saved) return saved;
    if (auto saved = Write(out / (b.name + ".rendered.txt"), task.encoded.rendered); !saved)
      return saved;
    if (comma) receipt += ',';
    comma = true;
    receipt += std::format(
        "{{\"name\":{},\"prompt_tokens\":{},\"capacity\":{},\"output_tokens\":{},"
        "\"qualitative\":{},\"hca_calls\":{},\"encoding_calls\":{},"
        "\"user_sha256\":{},\"rendered_sha256\":{},\"ids_file_sha256\":{},"
        "\"ids_sha256\":{},\"rubric\":{},\"answers\":[",
        Quoted(b.name), b.prompt_tokens, b.capacity, b.output_tokens,
        b.kind == lc::AnswerKind::kQualitative, task.hca_calls, task.encoding_calls,
        Quoted(Digest(task.user)), Quoted(Digest(task.encoded.rendered)), Quoted(Digest(ids)),
        Quoted(IdDigest(task.encoded.ids)), Quoted(b.rubric));
    for (std::size_t i = 0; i < b.answers.size(); ++i)
      receipt += (i == 0 ? "" : ",") + Quoted(b.answers[i]);
    receipt += "],\"facts\":[";
    for (std::size_t i = 0; i < b.facts.size(); ++i) {
      const auto& span = task.spans[i];
      receipt += std::format(
          "{}{{\"text\":{},\"first_byte\":{},\"end_byte\":{},"
          "\"first_token\":{},\"end_token\":{}}}",
          i == 0 ? "" : ",", Quoted(b.facts[i].text), span.first_byte, span.end_byte,
          span.first_token, span.end_token);
    }
    receipt += "]}";
  }
  receipt += "]}\n";
  return Write(out / "receipt.json", receipt);
}

Result Score(const NativeCodec& codec, const fs::path& prepared, std::string_view prepared_sha,
             std::string_view name, const fs::path& summary, const fs::path& out) {
  auto raw = Read(prepared / "receipt.json", std::size_t{1} << 20U);
  if (!raw || Digest(*raw) != prepared_sha)
    return std::unexpected("prepared receipt identity differs");
  auto manifest = js::Parse(*raw);
  if (!manifest) return std::unexpected("malformed prepared receipt");
  const auto root = manifest->root();
  const auto equal = [&](std::string_view key, std::string_view expected) {
    const auto value = root.find(key);
    return value && value->is_string() && value->string() == expected;
  };
  const auto complete = root.find("complete");
  const auto cases = root.find("cases");
  const auto stops = root.find("stop_ids");
  if (!complete || !complete->boolean() || !equal("protocol", lc::kProtocol) ||
      !equal("artifact", lc::kArtifact) || !equal("metadata_sha256", codec.metadata_sha()) ||
      !equal("template_sha256", codec.template_sha()) || !cases || !cases->is_array() ||
      cases->size() != 4 || !stops || !stops->is_array() || stops->size() != codec.stops().size())
    return std::unexpected("prepared native metadata/protocol or complete task set differs");
  for (std::size_t i = 0; i < stops->size(); ++i)
    if (stops->at(i).int64() != codec.stops()[i]) return std::unexpected("prepared stops differ");
  const auto blueprints = lc::FixedCases();
  const auto found = std::ranges::find(blueprints, name, &lc::Blueprint::name);
  if (found == blueprints.end()) return std::unexpected("unknown preregistered case");
  const auto index = static_cast<std::size_t>(found - blueprints.begin());
  const auto entry = cases->at(index);
  const auto entry_name = entry.find("name");
  auto user = Read(prepared / (found->name + ".user.txt"), std::size_t{4} << 20U);
  auto rendered = Read(prepared / (found->name + ".rendered.txt"), std::size_t{4} << 20U);
  auto ids_file = Read(prepared / (found->name + ".ids.tsv"), std::size_t{2} << 20U);
  if (!entry_name || entry_name->string() != found->name || !user || !rendered || !ids_file)
    return std::unexpected("missing prepared case files");
  auto e = codec.Encode(*user);
  const auto checked_digest = [&](std::string_view field, std::string_view actual) {
    const auto value = entry.find(field);
    return value && value->string() == actual;
  };
  if (!e || e->rendered != *rendered || e->ids.size() != found->prompt_tokens ||
      !checked_digest("user_sha256", Digest(*user)) ||
      !checked_digest("rendered_sha256", Digest(*rendered)) ||
      !checked_digest("ids_file_sha256", Digest(*ids_file)) ||
      !checked_digest("ids_sha256", IdDigest(e->ids)))
    return std::unexpected("frozen case bytes or native tokenization differs");
  auto raw_summary = Read(summary, std::size_t{1} << 20U);
  if (!raw_summary) return std::unexpected(raw_summary.error());
  auto execution = js::Parse(*raw_summary);
  if (!execution) return std::unexpected("malformed native summary");
  const auto run = execution->root();
  const auto artifact = run.find("artifact");
  const auto context = run.find("context");
  const auto max_rows = run.find("max_rows");
  const auto compact = run.find("compact_experts");
  const auto frontier = run.find("frontier_head");
  const auto wide = run.find("wide_sparse");
  const auto exact = run.find("exact");
  const auto q2 = run.find("q2_d2r");
  const auto hca = run.find("ds4_hca");
  const auto calls = run.find("hca_tokentile_steps");
  const auto prompts = run.find("prompts");
  const auto run_stops = run.find("stop_ids");
  auto expected_calls = lc::HcaCalls(found->prompt_tokens);
  if (!expected_calls) return std::unexpected(expected_calls.error());
  if (!artifact || artifact->string() != lc::kArtifact || !context ||
      context->int64() != found->capacity || !max_rows || max_rows->int64() != lc::kChunkRows ||
      !compact || !compact->boolean() || !frontier || !frontier->boolean() || !wide ||
      !wide->boolean() || !exact || !exact->is_bool() || exact->boolean() || !q2 ||
      !q2->is_bool() || q2->boolean() || !hca || !hca->is_bool() || !calls ||
      calls->int64() != (hca->boolean() ? *expected_calls : 0) || !prompts ||
      !prompts->is_array() || prompts->size() != 1 || !run_stops || !run_stops->is_array() ||
      run_stops->size() != codec.stops().size())
    return std::unexpected("native arm scope, flags or actual HCA count differs");
  for (std::size_t i = 0; i < run_stops->size(); ++i)
    if (run_stops->at(i).int64() != codec.stops()[i])
      return std::unexpected("native arm did not use the authenticated stop IDs");
  const auto prompt = prompts->at(0);
  const auto prompt_name = prompt.find("name");
  const auto count = prompt.find("prompt_tokens");
  const auto argmax = prompt.find("argmax");
  const auto reason = prompt.find("stop_reason");
  const auto output_count = prompt.find("output_tokens");
  const auto decode_steps = prompt.find("decode_steps");
  if (!prompt_name || prompt_name->string() != found->name || !count ||
      count->int64() != found->prompt_tokens || !argmax || !argmax->is_array() ||
      argmax->size() == 0 || argmax->size() > found->output_tokens || !reason || !output_count ||
      !decode_steps || decode_steps->int64() != argmax->size() - 1)
    return std::unexpected("native output geometry is incomplete");
  std::vector<std::int32_t> output;
  for (std::size_t i = 0; i < argmax->size(); ++i) {
    const auto id = argmax->at(i).int64();
    if (!id || *id < 0 || *id >= 129280) return std::unexpected("invalid generated token");
    output.push_back(static_cast<std::int32_t>(*id));
  }
  const auto stop =
      std::ranges::find_if(output, [&](auto id) { return lc::IsStop(codec.stops(), id); });
  const bool eos = stop != output.end();
  if ((eos && stop + 1 != output.end()) || (!eos && output.size() != found->output_tokens) ||
      reason->string() != (eos ? "eos" : "length") ||
      output_count->int64() != output.size() - (eos ? 1U : 0U))
    return std::unexpected("native EOS/length accounting differs from authenticated stops");
  auto response = codec.Decode(std::span(output.begin(), stop));
  if (!response) return std::unexpected(response.error());
  std::optional<bool> accepted;
  if (found->kind != lc::AnswerKind::kQualitative) {
    auto exact_answer = lc::ExactAnswer(*found, *response, reason->string());
    if (!exact_answer) return std::unexpected(exact_answer.error());
    accepted = *exact_answer;
  }
  if (auto fresh = Fresh(out); !fresh) return fresh;
  if (auto saved = Write(out / "response.txt", *response); !saved) return saved;
  std::string_view accepted_json = "null";
  if (accepted) accepted_json = *accepted ? "true" : "false";
  const auto score = std::format(
      "{{\"complete\":true,\"task\":{},\"prepared_receipt_sha256\":{},"
      "\"native_summary_sha256\":{},\"response_sha256\":{},\"stop_reason\":{},"
      "\"output_tokens\":{},\"qualitative\":{},\"accepted\":{},\"rubric\":{},"
      "\"full_quality_gate\":false}}\n",
      Quoted(found->name), Quoted(prepared_sha), Quoted(Digest(*raw_summary)),
      Quoted(Digest(*response)), Quoted(reason->string()), output.size() - (eos ? 1U : 0U),
      found->kind == lc::AnswerKind::kQualitative, accepted_json, Quoted(found->rubric));
  return Write(out / "receipt.json", score);
}
}  // namespace

int main(int argc, char** argv) {
  const bool preparing = argc == 4 && std::string_view(argv[1]) == "prepare";
  const bool scoring = argc == 8 && std::string_view(argv[1]) == "score";
  if (!preparing && !scoring) {
    std::println(stderr,
                 "usage: llmp_long_context_task_prepare prepare ARTIFACT FRESH_OUT\n"
                 "       llmp_long_context_task_prepare score ARTIFACT PREPARED RECEIPT_SHA "
                 "CASE NATIVE_SUMMARY FRESH_OUT");
    return 2;
  }
  auto codec = OpenCodec(argv[2]);
  if (!codec) {
    std::println(stderr, "{}", codec.error());
    return 1;
  }
  auto result = argc == 4 ? Prepare(*codec, argv[3])
                          : Score(*codec, argv[3], argv[4], argv[5], argv[6], argv[7]);
  if (!result) {
    std::println(stderr, "{}", result.error());
    return 1;
  }
  return 0;
}
