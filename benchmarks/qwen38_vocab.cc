// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "qwen38_vocab.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <format>
#include <fstream>
#include <iterator>
#include <optional>
#include <set>
#include <system_error>
#include <utility>

#include "base/json.h"
#include "base/sha256.h"
#include "chat/chat.h"
#include "model/qwen38.h"
#include "tokenizer/hf.h"
#include "tokenizer/tokenizer.h"

namespace jitllm::benchmarks::draft_vocab {
namespace {

auto Error(std::string text) { return std::unexpected(std::move(text)); }

bool SafeName(std::string_view text) {
  return !text.empty() && text.size() <= 64 && std::ranges::all_of(text, [](char c) {
    return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-';
  });
}

std::expected<std::vector<std::int32_t>, std::string> Ids(base::json::Value value,
                                                          std::size_t limit) {
  if (!value.is_array() || value.size() == 0 || value.size() > limit) {
    return Error("empty or oversized frozen vocabulary-study IDs");
  }
  std::vector<std::int32_t> ids;
  ids.reserve(value.size());
  for (std::size_t i = 0; i < value.size(); ++i) {
    const auto id = value.at(i).int64();
    if (!id || *id < 0 || *id >= 248320) {
      return Error("vocabulary-study ID is not an in-range integer");
    }
    ids.push_back(static_cast<std::int32_t>(*id));
  }
  return ids;
}

}  // namespace

std::expected<void, std::string> CheckExamples(std::span<const Example> examples,
                                               std::uint32_t context, std::uint32_t vocab) {
  constexpr std::array<std::string_view, 4> kDomains = {"code", "prose", "instruction",
                                                        "arithmetic"};
  constexpr std::array<std::uint32_t, kAnchors> kOffsets = {0, 8, 16, 24};
  if (examples.size() != 8 || context == 0 || context > kContext || vocab != 248320) {
    return Error("vocabulary study requires all eight bounded domain/depth cells");
  }
  std::set<std::pair<std::string, std::uint32_t>> cells;
  std::set<std::string> names;
  for (const Example& e : examples) {
    if (!SafeName(e.id) || !names.insert(e.id).second ||
        (e.split != "calibration" && e.split != "held_out") || e.split != examples.front().split ||
        std::ranges::find(kDomains, e.domain) == kDomains.end() ||
        (e.nominal_tokens != 8192 && e.nominal_tokens != 32768) ||
        !cells.emplace(e.domain, e.nominal_tokens).second) {
      return Error("vocabulary study has a duplicate or unknown cell/identity");
    }
    if (e.prompt.size() > e.nominal_tokens || e.prompt.size() + 128 < e.nominal_tokens ||
        e.stable_boundary == 0 || e.stable_boundary >= e.prompt.size() ||
        e.continuation.size() < 28 || e.continuation.size() > 256 || e.anchors.size() != kAnchors ||
        !std::ranges::equal(e.anchors, kOffsets) ||
        e.prompt.size() + e.continuation.size() + kDepth + 1 > context) {
      return Error("vocabulary-study prompt/history/anchor extent is invalid");
    }
    for (const auto* ids : {&e.prompt, &e.continuation}) {
      if (!std::ranges::all_of(
              *ids, [vocab](std::int32_t id) { return id >= 0 && std::cmp_less(id, vocab); })) {
        return Error("vocabulary-study IDs are outside the target vocabulary");
      }
    }
  }
  return {};
}

std::expected<std::vector<Example>, std::string> ReadExamples(std::string_view json,
                                                              std::uint32_t context,
                                                              std::uint32_t vocab) {
  auto parsed = base::json::Parse(json, {.max_bytes = 8U << 20U,
                                         .max_depth = 8,
                                         .max_values = 1U << 20U,
                                         .max_string_bytes = 4U << 20U});
  if (!parsed) {
    return Error(parsed.error().ToString());
  }
  const auto root = parsed->root();
  const auto format = root.find("format");
  const auto split = root.find("split");
  const auto list = root.find("examples");
  const auto signal = root.find("target_signal");
  const auto context_value = root.find("context_capacity");
  const auto chunk = root.find("prefill_chunk");
  const auto depth = root.find("draft_depth");
  if (!format || format->string() != "jitllm-qwen-own-vocab-prepared-v1" || !split ||
      (split->string() != "calibration" && split->string() != "held_out") || !list ||
      !list->is_array() || list->size() != 8 || !signal ||
      signal->string() != "native_control_natural_argmax" || !context_value ||
      context_value->int64() != kContext || context != kContext || !chunk ||
      chunk->int64() != 8192 || !depth || depth->int64() != kDepth) {
    return Error("vocabulary-study prepared manifest has an invalid contract");
  }
  std::vector<Example> out;
  for (std::size_t i = 0; i < list->size(); ++i) {
    const auto item = list->at(i);
    const auto id = item.find("id");
    const auto domain = item.find("domain");
    const auto nominal = item.find("nominal_tokens");
    const auto boundary = item.find("stable_boundary");
    const auto nominal_value = nominal ? nominal->int64() : std::nullopt;
    const auto boundary_value = boundary ? boundary->int64() : std::nullopt;
    const auto prompt = item.find("prompt_ids");
    const auto continuation = item.find("continuation_ids");
    const auto anchors = item.find("anchors");
    if (!id || !domain || !nominal || !boundary || !prompt || !continuation || !anchors ||
        !nominal_value || !boundary_value || *nominal_value < 1 || *nominal_value > 32768 ||
        *boundary_value < 1 || *boundary_value > 32768 || !anchors->is_array() ||
        anchors->size() != kAnchors) {
      return Error("vocabulary-study example is missing bounded fields");
    }
    auto p = Ids(*prompt, 32768);
    auto c = Ids(*continuation, 256);
    if (!p || !c) {
      return Error(!p ? p.error() : c.error());
    }
    Example e{.id = std::string(id->string()),
              .domain = std::string(domain->string()),
              .split = std::string(split->string()),
              .nominal_tokens = static_cast<std::uint32_t>(*nominal_value),
              .stable_boundary = static_cast<std::uint32_t>(*boundary_value),
              .prompt = std::move(*p),
              .continuation = std::move(*c),
              .anchors = {}};
    for (std::size_t a = 0; a < anchors->size(); ++a) {
      const auto offset = anchors->at(a).int64();
      if (!offset || *offset < 0 || *offset > 24) {
        return Error("vocabulary-study anchor is not a bounded integer");
      }
      e.anchors.push_back(static_cast<std::uint32_t>(*offset));
    }
    out.push_back(std::move(e));
  }
  if (auto checked = CheckExamples(out, context, vocab); !checked) {
    return Error(checked.error());
  }
  return out;
}

namespace {

std::expected<std::string, std::string> ReadBounded(const std::filesystem::path& path,
                                                    std::size_t maximum) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  const auto size = static_cast<std::streamoff>(file.tellg());
  if (!file || size < 0 || std::cmp_greater(size, maximum)) {
    return Error("unreadable or oversized vocabulary preparation input: " + path.string());
  }
  file.seekg(0);
  std::string text(static_cast<std::size_t>(size), '\0');
  file.read(text.data(), static_cast<std::streamsize>(text.size()));
  if (!file) {
    return Error("reading vocabulary preparation input: " + path.string());
  }
  return text;
}

std::string Quoted(std::string_view text) {
  std::string out;
  base::json::AppendQuoted(text, out);
  return out;
}

std::string Digest(std::string_view text) {
  base::Sha256 hash;
  hash.Update(text);
  return base::ToHex(hash.Finish());
}

template <class T>
std::string Numbers(std::span<const T> values) {
  std::string out;
  for (const T value : values) {
    out += std::format("{}{}", out.empty() ? "" : ",", value);
  }
  return "[" + out + "]";
}

std::string Replace(std::string text, std::string_view needle, std::uint32_t number) {
  const std::string value = std::to_string(number);
  std::size_t at = 0;
  while ((at = text.find(needle, at)) != std::string::npos) {
    text.replace(at, needle.size(), value);
    at += value.size();
  }
  return text;
}

struct RenderedExample {
  std::string text;
  std::vector<std::int32_t> ids;
  std::uint32_t boundary = 0;
};

std::expected<RenderedExample, std::string> Render(std::string content,
                                                   const tokenizer::Tokenizer& tokenizer) {
  chat::Conversation conversation;
  conversation.messages.push_back({.role = chat::Role::kUser,
                                   .content = std::move(content),
                                   .reasoning_content = std::nullopt,
                                   .tool_calls = {}});
  auto rendered = chat::RenderQwen38(conversation);
  if (!rendered) {
    return Error(rendered.error().ToString());
  }
  RenderedExample out;
  std::vector<std::size_t> spans;
  if (auto encoded =
          tokenizer.EncodeMarked(rendered->text, rendered->specials, {}, out.ids, &spans);
      !encoded) {
    return Error(encoded.error().ToString());
  }
  for (const chat::Boundary& boundary : rendered->boundaries) {
    if (boundary.kind == chat::BoundaryKind::kGenerationPrompt) {
      for (std::size_t i = 0; i < rendered->specials.size(); ++i) {
        if (rendered->specials[i].offset == boundary.offset && spans[i] < out.ids.size()) {
          out.boundary = static_cast<std::uint32_t>(spans[i]);
        }
      }
    }
  }
  out.text = std::move(rendered->text);
  return out;
}

std::expected<std::vector<std::int32_t>, std::string> Stops(base::json::Value metadata,
                                                            const chat::Template& renderer,
                                                            const tokenizer::Tokenizer& tokenizer) {
  auto native = chat::StopTokens(renderer.stop, tokenizer);
  if (!native) {
    return Error(native.error().ToString());
  }
  std::vector<std::int32_t> out = std::move(*native);
  if (const auto eos = tokenizer.eos(); eos) {
    out.push_back(*eos);
  }
  const auto eos = metadata.find("eos_token_id");
  if (!eos) {
    return Error("target stop metadata must contain eos_token_id");
  }
  for (const auto value : {eos, metadata.find("stop_token_ids")}) {
    if (!value) {
      continue;
    }
    const auto add = [&](base::json::Value id) -> bool {
      const auto integer = id.int64();
      if (!integer || *integer < 0 || *integer >= 248320) {
        return false;
      }
      out.push_back(static_cast<std::int32_t>(*integer));
      return true;
    };
    if (value->is_array()) {
      if (value->size() > 16) {
        return Error("too many target stop IDs");
      }
      for (std::size_t i = 0; i < value->size(); ++i) {
        if (!add(value->at(i))) {
          return Error("invalid target stop ID");
        }
      }
    } else if (!add(*value)) {
      return Error("invalid target stop ID");
    }
  }
  std::ranges::sort(out);
  const auto duplicate = std::ranges::unique(out);
  out.erase(duplicate.begin(), duplicate.end());
  if (out.empty() || out.size() > 16) {
    return Error("invalid complete target stop set");
  }
  return out;
}

}  // namespace

std::expected<void, std::string> Prepare(const Preparation& paths) {
  auto source_text = ReadBounded(paths.source, 64U << 10U);
  auto tokenizer_text = ReadBounded(paths.tokenizer, 64U << 20U);
  auto template_text = ReadBounded(paths.chat_template, 256U << 10U);
  auto metadata_text = ReadBounded(paths.stop_metadata, 64U << 10U);
  if (!source_text) return Error(source_text.error());
  if (!tokenizer_text) return Error(tokenizer_text.error());
  if (!template_text) return Error(template_text.error());
  if (!metadata_text) return Error(metadata_text.error());
  auto source = base::json::Parse(*source_text);
  auto metadata = base::json::Parse(*metadata_text);
  auto spec = tokenizer::ReadHfTokenizer(*tokenizer_text);
  auto renderer = chat::FindTemplateForText(*template_text);
  if (!source || !metadata || !spec || !renderer) {
    return Error("vocabulary preparation JSON, tokenizer or native template is invalid");
  }
  if ((*renderer)->name != "qwen3.8-flash-next") {
    return Error("vocabulary preparation requires the pinned Qwen3.8 renderer");
  }
  auto encoder = tokenizer::Tokenizer::Create(std::move(*spec));
  if (!encoder) {
    return Error(encoder.error().ToString());
  }
  auto stops = Stops(metadata->root(), **renderer, *encoder);
  if (!stops) {
    return Error(stops.error());
  }
  const auto root = source->root();
  const auto field = [&](std::string_view name) {
    const auto value = root.find(name);
    return value ? value->string() : std::string_view{};
  };
  const auto examples = root.find("examples");
  const auto background = root.find("background");
  if (field("format") != "jitllm-qwen-own-vocab-source-v1" ||
      field("target_signal") != "native_control_natural_argmax" ||
      field("license") != "Apache-2.0" ||
      field("target_artifact") !=
          "c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93" ||
      !examples || !examples->is_array() || examples->size() != 8 || !background) {
    return Error("independent source manifest does not match the frozen study contract");
  }
  const auto integer = [&](base::json::Value object, std::string_view name, std::int64_t value) {
    const auto found = object.find(name);
    return found && found->int64() == value;
  };
  const auto construction = root.find("construction");
  const auto anchors = root.find("anchors");
  const auto depths = root.find("nominal_prompt_tokens");
  if (!integer(root, "context_capacity", kContext) || !integer(root, "prefill_chunk", 8192) ||
      !integer(root, "draft_depth", kDepth) || !integer(*background, "maximum_records", 4096) ||
      !construction || !integer(*construction, "rows", 47172) ||
      !construction->find("proposal_weight") ||
      construction->find("proposal_weight")->string() != "3/4" ||
      !construction->find("target_weight") ||
      construction->find("target_weight")->string() != "1/4" || !anchors || !anchors->is_array() ||
      anchors->size() != 4 || !depths || !depths->is_array() || depths->size() != 2 ||
      depths->at(0).int64() != 8192 || depths->at(1).int64() != 32768) {
    return Error("independent manifest changed the fixed construction or collection bounds");
  }
  for (std::size_t i = 0; i < 4; ++i) {
    if (anchors->at(i).int64() != static_cast<std::int64_t>(i * 8)) {
      return Error("independent manifest changed a fixed anchor");
    }
  }
  const auto header = background->find("header");
  const auto footer = background->find("footer");
  if (!header || !footer || header->string().empty() || footer->string().empty()) {
    return Error("independent background delimiters are missing");
  }
  std::error_code directory_error;
  std::filesystem::create_directories(paths.output, directory_error);
  if (directory_error) {
    return Error("creating prepared vocabulary manifest directory");
  }
  for (std::string_view split : {"calibration", "held_out"}) {
    if (std::filesystem::exists(paths.output / std::format("{}.json", split), directory_error) ||
        directory_error) {
      return Error("prepared manifests are immutable; a fresh output directory is required");
    }
  }
  const auto state = model::Qwen38State(model::Qwen38Flash(), kContext, 8192, false);
  if (!state) {
    return Error("CPU state layout planning failed");
  }
  const auto mtp = model::Qwen38MtpStateOf(model::Qwen38Flash(), *state);
  if (!mtp) {
    return Error("CPU state layout planning failed");
  }
  for (std::string_view split : {"calibration", "held_out"}) {
    std::vector<Example> prepared;
    std::string records;
    for (std::size_t i = 0; i < examples->size(); ++i) {
      const auto entry = examples->at(i);
      const auto part = entry.find("split");
      if (!part || part->string() != split) {
        continue;
      }
      const auto id = entry.find("id");
      const auto domain = entry.find("domain");
      const auto seed = entry.find("seed");
      const auto request = entry.find("request");
      const auto history = entry.find("common_history");
      const auto line = domain ? background->find(domain->string()) : std::nullopt;
      if (!id || !domain || !seed || !seed->int64() || *seed->int64() < 0 ||
          *seed->int64() > 100003 || !request || !history || !line || request->string().empty() ||
          history->string().empty() || line->string().empty() || !SafeName(id->string()) ||
          line->string().size() > 512) {
        return Error("independent example has invalid fields");
      }
      const auto seed_value = static_cast<std::uint32_t>(*seed->int64());
      std::string archive;
      std::vector<std::size_t> ends = {0};
      for (std::uint32_t n = 0; n < 4096; ++n) {
        auto text = Replace(std::string(line->string()), "{seed}", seed_value);
        text = Replace(std::move(text), "{record}", n);
        text = Replace(std::move(text), "{value}", ((seed_value * 17) + (n * 29)) % 100003);
        archive += text + '\n';
        ends.push_back(archive.size());
      }
      std::vector<std::int32_t> continuation;
      if (auto encoded = encoder->Encode(history->string(), {}, continuation); !encoded) {
        return Error(encoded.error().ToString());
      }
      const auto render = [&](std::size_t count) {
        return Render(std::string(header->string()) + "\n" + archive.substr(0, ends[count]) +
                          std::string(footer->string()) + "\n" + std::string(request->string()),
                      *encoder);
      };
      for (const std::uint32_t nominal : {8192U, 32768U}) {
        std::size_t low = 0;
        std::size_t high = 4096;
        while (low < high) {
          const std::size_t middle = low + ((high - low + 1) / 2);
          auto trial = render(middle);
          if (!trial) {
            return Error(trial.error());
          }
          if (trial->ids.size() <= nominal) {
            low = middle;
          } else {
            high = middle - 1;
          }
        }
        auto chosen = render(low);
        if (!chosen) {
          return Error(chosen.error());
        }
        if (low < 4096) {
          auto next = render(low + 1);
          if (!next || next->ids.size() <= nominal) {
            return Error("whole-record prompt boundary did not converge");
          }
        }
        Example example{.id = std::format("{}-{}", id->string(), nominal),
                        .domain = std::string(domain->string()),
                        .split = std::string(split),
                        .nominal_tokens = nominal,
                        .stable_boundary = chosen->boundary,
                        .prompt = std::move(chosen->ids),
                        .continuation = continuation,
                        .anchors = {0, 8, 16, 24}};
        records += std::format(
            R"({}{{"id":{},"domain":{},"nominal_tokens":{},"stable_boundary":{},"background_records":{},"rendered_text":{},"common_history_text":{},"prompt_ids":{},"continuation_ids":{},"anchors":[0,8,16,24]}})",
            records.empty() ? "" : ",", Quoted(example.id), Quoted(example.domain), nominal,
            example.stable_boundary, low, Quoted(chosen->text), Quoted(history->string()),
            Numbers(std::span<const std::int32_t>(example.prompt)),
            Numbers(std::span<const std::int32_t>(continuation)));
        prepared.push_back(std::move(example));
      }
    }
    if (auto valid = CheckExamples(prepared, kContext, 248320); !valid) {
      return valid;
    }
    const std::string text = std::format(
        R"({{"format":"jitllm-qwen-own-vocab-prepared-v1","split":"{}","target_signal":"native_control_natural_argmax","source_sha256":"{}","tokenizer_sha256":"{}","template_sha256":"{}","stop_metadata_sha256":"{}","stop_token_ids":{},"context_capacity":33792,"prefill_chunk":8192,"draft_depth":3,"target_state_capacity_bytes":{},"mtp_state_capacity_bytes":{},"model_loaded":false,"cuda_initialized":false,"examples":[{}]}})",
        split, Digest(*source_text), Digest(*tokenizer_text), Digest(*template_text),
        Digest(*metadata_text), Numbers(std::span<const std::int32_t>(*stops)), state->bytes,
        mtp->bytes, records);
    std::ofstream file(paths.output / std::format("{}.json", split),
                       std::ios::binary | std::ios::noreplace);
    file << text << '\n';
    file.close();
    if (!file.good()) {
      return Error("writing frozen vocabulary input manifest");
    }
  }
  return {};
}

}  // namespace jitllm::benchmarks::draft_vocab
