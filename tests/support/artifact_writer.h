// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A test-only writer of v1 prepared artifacts and compositions
// (docs/artifact-format.md, "Version 1"; D-113), for the reader's tests and
// for tests that need a small v1 artifact. Not the importer (N4's writer):
// it takes the layout it is given, fills tensors with deterministic bytes
// and writes the documents canonically, so that tests can then edit the
// manifest or index (as JSON trees or as text) and publish the result
// under its manifest's digest.
//
// A plan names the groups in file order. A resource group holds resources;
// an expert layer holds expert arrays and becomes `expert_count`
// consecutive expert groups, slice e of each array in group e at the same
// offset. Members are placed at the given offset or packed after the
// previous one's readable end at 256 bytes; a group's used bytes end at
// its last readable range. Shards break where a group asks for a new one.

#ifndef LLMP_TESTS_SUPPORT_ARTIFACT_WRITER_H_
#define LLMP_TESTS_SUPPORT_ARTIFACT_WRITER_H_

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "artifact/layout.h"
#include "format/layout.h"
#include "format/tensor_format.h"

namespace llmp::test {

// A mutable JSON tree written canonically: sorted keys, no whitespace,
// integers and escape-free strings only. Raw holds JSON text written as is.
class Json {
 public:
  using Array = std::vector<Json>;
  using Object = std::map<std::string, Json, std::less<>>;
  struct Raw {
    std::string text;
  };

  Json() = default;
  Json(std::nullptr_t /*null*/) {}  // NOLINT(google-explicit-constructor)
  Json(bool b) : v_(b) {}           // NOLINT(google-explicit-constructor)
  template <std::integral T>
    requires(!std::same_as<T, bool>)
  Json(T n) : v_(static_cast<std::int64_t>(n)) {}   // NOLINT(google-explicit-constructor)
  Json(std::string s) : v_(std::move(s)) {}         // NOLINT(google-explicit-constructor)
  Json(const char* s) : v_(std::string(s)) {}       // NOLINT(google-explicit-constructor)
  Json(std::string_view s) : v_(std::string(s)) {}  // NOLINT(google-explicit-constructor)
  Json(Array a) : v_(std::move(a)) {}               // NOLINT(google-explicit-constructor)
  Json(Object o) : v_(std::move(o)) {}              // NOLINT(google-explicit-constructor)
  Json(Raw r) : v_(std::move(r)) {}                 // NOLINT(google-explicit-constructor)

  // An object's member, made (null) when missing; the value becomes an
  // object first if it is not one.
  Json& operator[](std::string_view key);
  // An array's element.
  Json& at(std::size_t i);
  Array& array();
  Object& object();
  void Erase(std::string_view key);

  void Write(std::string& out) const;
  // The canonical document: Write's text and one newline.
  std::string Text() const;

 private:
  std::variant<std::nullptr_t, bool, std::int64_t, std::string, Array, Object, Raw> v_ = nullptr;
};

// The canonical JSON of a format (artifact/format_json.h AppendFormatJson).
Json FormatJson(const format::TensorFormat& f);

struct V1Resource {
  std::string name{};
  std::vector<std::string> aliases{};  // tied roles, sorted; a dedupe-identical each
  format::TensorFormat format{};
  format::PartRole part = format::PartRole::kData;
  bool rows = false;
  std::optional<std::uint64_t> offset{};  // in its group; default packed
};

struct V1Array {
  std::string name{};
  format::TensorFormat format{};  // one slice's
  format::PartRole part = format::PartRole::kData;
  std::optional<std::uint64_t> offset{};  // in each expert group; default packed
};

struct V1Group {
  artifact::GroupKind kind = artifact::GroupKind::kLayer;
  std::optional<std::uint32_t> layer{};
  std::vector<V1Resource> resources{};  // a resource group's
  std::vector<V1Array> arrays{};        // an expert layer's (kind kExpert, layer set)
  bool new_shard = false;               // start a new shard here
};

struct V1Plan {
  std::string architecture = "toy";
  std::string container = "gguf";
  std::uint32_t expert_count = 0;
  std::string importer = "llmp-import-test";
  std::string importer_version = "1";
  std::string importer_build = "test";
  // Kept metadata (meta/<name>), each also recorded as a source, and other
  // recorded sources (a checkpoint's own files), by name.
  std::vector<std::pair<std::string, std::string>> meta{};
  std::vector<std::pair<std::string, std::string>> sources = {{"toy.gguf", "toy source bytes"}};
  std::optional<std::string> spec{};  // spec.json's bytes
  std::vector<V1Group> groups{};
  std::vector<std::pair<std::string, std::vector<std::string>>> folds{};  // (fold, tensors)
  std::vector<std::pair<std::string, std::vector<std::string>>> source_folds{};
  std::vector<std::pair<format::Layout, std::vector<std::string>>> relayouts{};
};

// An artifact's documents and files, rendered and not yet published: tests
// edit `manifest` and `index` (trees, rendered at Publish) or, through
// Publish's hooks, their text. The manifest's `files` list is filled in
// at Publish from the files' bytes unless `keep_files` is set.
struct V1Files {
  Json manifest{};
  Json index{};
  std::optional<std::string> spec{};
  std::vector<std::string> shards{};  // whole shard files, data/NNNNN.safetensors in order
  std::vector<std::pair<std::string, std::string>> meta{};
  bool keep_files = false;
};

V1Files Render(const V1Plan& plan);

struct PublishHooks {
  std::function<void(std::string&)> index{};     // edits index.json's text
  std::function<void(std::string&)> manifest{};  // edits manifest.json's text, before the ID
};

// Writes the files into store/<SHA-256 of manifest.json> and returns that
// directory (replacing one already there).
std::filesystem::path Publish(const std::filesystem::path& store, const V1Files& files,
                              const PublishHooks& hooks = {});

// One entry of a shard header (a resource, an expert slice or a pad):
// its name, container dtype and shape, and its [start, end) in the data
// section.
struct ShardEntry {
  std::string name{};
  std::string dtype{};
  std::vector<std::uint64_t> shape{};
  std::uint64_t start = 0;
  std::uint64_t end = 0;
};

// The header bytes the reader rebuilds for a shard: the 8-byte length,
// the JSON with `metadata` (`{"__metadata__":...}` without its closing
// brace) and the entries in order, then spaces to the 4 KiB-aligned data
// offset. Exposed for tests that edit an index.
std::string ShardHeader(std::string_view metadata, const std::vector<ShardEntry>& entries);

// A v1 composition: components (role, artifact ID, architecture) in role
// order, kept files (each a source) and an optional spec.
struct CompositionPlan {
  std::string architecture = "ToyPipeline";
  std::vector<std::tuple<std::string, std::string, std::string>> components{};
  std::vector<std::pair<std::string, std::string>> meta{};
  std::optional<std::string> spec{};
};

struct CompositionFiles {
  Json manifest{};
  std::optional<std::string> spec{};
  std::vector<std::pair<std::string, std::string>> meta{};
  bool keep_files = false;
};

CompositionFiles Render(const CompositionPlan& plan);
std::filesystem::path Publish(const std::filesystem::path& store, const CompositionFiles& files,
                              const PublishHooks& hooks = {});

}  // namespace llmp::test

#endif  // LLMP_TESTS_SUPPORT_ARTIFACT_WRITER_H_
