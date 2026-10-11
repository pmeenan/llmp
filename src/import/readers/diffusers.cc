// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "import/readers/diffusers.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "base/json.h"
#include "import/checkpoint.h"
#include "import/error.h"
#include "import/files.h"
#include "import/reader.h"
#include "import/readers/safetensors.h"

namespace llmp::import {
namespace {

namespace json = base::json;

constexpr std::string_view kModelIndexName = "model_index.json";
constexpr std::string_view kConfigName = "config.json";
constexpr std::size_t kMaxRole = 64;
constexpr std::size_t kMaxFileName = 255;

std::unexpected<Error> Fail(Rule rule, std::string_view reason, std::uint64_t file = kNoItem,
                            ItemKind kind = ItemKind::kNone, std::uint64_t item = kNoItem) {
  return std::unexpected(
      Error{.rule = rule, .reason = reason, .file = file, .kind = kind, .item = item});
}

// [a-z][a-z0-9_]{0,63}.
bool Role(std::string_view s) {
  if (s.empty() || s.size() > kMaxRole || s.front() < 'a' || s.front() > 'z') {
    return false;
  }
  return std::ranges::all_of(
      s, [](char c) { return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_'; });
}

// A [library, class] pair: a list of two strings.
bool NamesComponent(json::Value v) {
  return v.is_array() && v.size() == 2 && v.at(0).is_string() && v.at(1).is_string();
}

// Whether `path` lies strictly beneath `root` (both resolved, absolute).
bool Beneath(const std::filesystem::path& path, const std::filesystem::path& root) {
  const std::filesystem::path relative = path.lexically_relative(root);
  return !relative.empty() && *relative.begin() != "." && *relative.begin() != "..";
}

// The regular files beneath a component's directory, by name.
std::expected<std::vector<std::string>, Error> FileNames(const std::filesystem::path& dir) {
  std::error_code error;
  std::filesystem::directory_iterator it(dir, error);
  if (error) {
    return Fail(Rule::kIo, "cannot list a pipeline component's directory");
  }
  std::vector<std::string> names;
  std::size_t entries = 0;
  while (it != std::filesystem::directory_iterator()) {
    if (++entries > kMaxPipelineDirectoryEntries) {
      return Fail(Rule::kBounds, "a pipeline component's directory of more entries than the cap");
    }
    std::error_code status_error;
    const std::filesystem::file_status status = it->status(status_error);  // links followed
    if (status.type() == std::filesystem::file_type::not_found) {
      // Listed but not there through its links: a link to nothing (a
      // cache's blob not downloaded). Leaving the file out would make a
      // pipeline without it. (status() reports ENOENT as an error too.)
      return Fail(Rule::kFileType, "a pipeline file is a link to nothing");
    }
    if (!status_error && std::filesystem::is_regular_file(status)) {
      std::string name = it->path().filename().string();
      if (name.empty() || name.size() > kMaxFileName || json::FirstInvalidUtf8(name) ||
          !NoControlCharacters(name)) {
        return Fail(Rule::kFileType,
                    "a pipeline file whose name is not UTF-8 without control characters");
      }
      names.push_back(std::move(name));
    }
    it.increment(error);
    if (error) {
      return Fail(Rule::kIo, "cannot list a pipeline component's directory");
    }
  }
  std::ranges::sort(names);
  return names;
}

}  // namespace

std::expected<ModelIndex, Error> ParseModelIndex(std::string_view text) {
  if (text.size() > kMaxModelIndexBytes) {
    return Fail(Rule::kBounds, "a model_index.json larger than its cap");
  }
  const auto doc =
      json::Parse(text, {.max_bytes = static_cast<std::size_t>(kMaxModelIndexBytes),
                         .max_depth = kMaxConfigDepth,
                         .max_values = std::size_t{1} << 16U,
                         .max_string_bytes = static_cast<std::size_t>(kMaxModelIndexBytes)});
  if (!doc) {
    return Fail(Rule::kConfig, "model_index.json is not valid JSON within the reader's limits",
                kNoItem, ItemKind::kByte, doc.error().offset);
  }
  const json::Value root = doc->root();
  if (!root.is_object()) {
    return Fail(Rule::kConfig, "model_index.json is not an object");
  }
  const auto name = root.find("_class_name");
  if (!name || !name->is_string() || name->string().empty() ||
      name->string().size() > kMaxArchitecture || !NoControlCharacters(name->string())) {
    return Fail(Rule::kConfig, "model_index.json has no _class_name string of 1 to 200 bytes");
  }
  ModelIndex out{.architecture = std::string(name->string())};
  for (std::size_t i = 0; i < root.size(); ++i) {
    const std::string_view key = root.key(i);
    if (key.starts_with('_') || !NamesComponent(root.member(i))) {
      continue;
    }
    if (!Role(key)) {
      return Fail(Rule::kConfig, "a pipeline component whose name is not [a-z][a-z0-9_]{0,63}",
                  kNoItem, ItemKind::kKey, i);
    }
    if (out.roles.size() >= kMaxPipelineEntries) {
      return Fail(Rule::kBounds, "more pipeline components than the reader's cap");
    }
    out.roles.emplace_back(key);
  }
  std::ranges::sort(out.roles);
  auto metadata = FlattenMetadata(root);
  if (!metadata) {
    return std::unexpected(metadata.error());
  }
  out.metadata = std::move(*metadata);
  return out;
}

std::expected<Checkpoint, Error> ReadDiffusers(const std::filesystem::path& directory) {
  auto root = CheckpointDirectory(directory);
  if (!root) {
    return std::unexpected(root.error());
  }
  auto index_file = OpenCheckpointFile(*root / kModelIndexName, 0);
  if (!index_file) {
    return std::unexpected(index_file.error());
  }
  auto text = ReadDocument(*index_file, kMaxModelIndexBytes, 0);
  if (!text) {
    return std::unexpected(text.error());
  }
  auto index = ParseModelIndex(*text);
  if (!index) {
    return std::unexpected(index.error());
  }
  Checkpoint c{.container = "diffusers", .architecture = std::move(index->architecture)};
  c.metadata = std::move(index->metadata);
  c.files.push_back({.name = std::string(kModelIndexName),
                     .path = index_file->path,
                     .bytes = index_file->bytes,
                     .device = index_file->device,
                     .inode = index_file->inode,
                     .parsed = {index_file->bytes, base::Sha256().Update(*text).Finish()}});
  std::uint64_t kept = 0;
  for (std::size_t r = 0; r < index->roles.size(); ++r) {
    const std::string& role = index->roles[r];
    // Whatever goes wrong inside a component names it (by its role's index).
    const auto in_component = [r](Error e) {
      e.component = r;
      return std::unexpected(e);
    };
    auto dir = CheckpointDirectory(*root / role);
    if (!dir) {
      return in_component(dir.error());
    }
    // A component's directory, its links resolved, lies beneath the
    // pipeline's (and is not the pipeline's own): a link out of it is an
    // escape from the checkpoint.
    if (!Beneath(*dir, *root)) {
      return in_component(Error{.rule = Rule::kEscape,
                                .reason = "a pipeline component's directory outside the pipeline"});
    }
    auto config = OpenCheckpointFileIfPresent(*dir / kConfigName, kNoItem);
    if (!config) {
      return in_component(config.error());
    }
    if (config->has_value()) {
      // Weights: a checkpoint of its own.
      if (c.components.size() >= kMaxPipelineComponents) {
        return in_component(Error{.rule = Rule::kBounds,
                                  .reason = "more weight components than a composition holds"});
      }
      auto component = ReadComponentCheckpoint(*dir);
      if (!component) {
        return in_component(component.error());
      }
      component->role = role;
      c.components.push_back(std::move(*component));
      continue;
    }
    // A scheduler's, a tokenizer's, a processor's files, kept verbatim.
    auto names = FileNames(*dir);
    if (!names) {
      return in_component(names.error());
    }
    for (const std::string& name : *names) {
      const std::uint64_t file = c.files.size();
      if (file > kMaxPipelineFiles) {  // model_index.json is file 0
        return in_component(Error{.rule = Rule::kBounds,
                                  .reason = "more pipeline files than a composition keeps",
                                  .file = file});
      }
      // Kept verbatim, so it must resolve inside the pipeline (files.h);
      // one that resolves elsewhere is left out, and said so.
      if (!ResolvesInside(*root, *dir / name)) {
        c.skipped.push_back(
            {.name = role + "/" + name, .reason = "resolves outside the checkpoint"});
        continue;
      }
      auto f = OpenCheckpointFile(*dir / name, file);
      if (!f) {
        return in_component(f.error());
      }
      if (f->bytes > kMaxPipelineFileBytes || f->bytes > kMaxPipelineFilesBytes - kept) {
        return in_component(Error{.rule = Rule::kBounds,
                                  .reason = "pipeline files beyond what a composition keeps",
                                  .file = file});
      }
      kept += f->bytes;
      c.files.push_back({.name = role + "/" + name,
                         .path = f->path,
                         .bytes = f->bytes,
                         .device = f->device,
                         .inode = f->inode});
    }
  }
  if (c.components.empty()) {
    return Fail(Rule::kConfig, "a pipeline with no component that holds weights");
  }
  return c;
}

}  // namespace llmp::import
