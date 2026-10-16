// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "import/archive_input.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <format>
#include <map>
#include <set>
#include <span>
#include <string_view>

#include "base/json.h"
#include "chat/pycase.h"
#include "platform/sockets.h"

namespace llmp::import {
namespace {
namespace fs = std::filesystem;
namespace json = base::json;
constexpr std::uint64_t kManifestCap = std::uint64_t{16} << 20U;
// A safetensors index, bounded as its reader bounds it (readers/safetensors.h).
constexpr std::uint64_t kIndexCap = std::uint64_t{64} << 20U;
std::unexpected<Error> Bad(std::string_view reason, Rule rule = Rule::kConfig) {
  return std::unexpected(Error{.rule = rule, .reason = reason});
}
std::string Q(std::string_view value) {
  std::string out;
  json::AppendQuoted(value, out);
  return out;
}
bool Hex(std::string_view text, std::size_t size) {
  return text.size() == size && std::ranges::all_of(text, [](char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
         });
}
base::Sha256Digest Digest(std::string_view text) {
  const auto nibble = [](char c) { return c <= '9' ? c - '0' : c - 'a' + 10; };
  base::Sha256Digest out{};
  for (std::size_t i = 0; i < out.size(); ++i)
    out[i] = static_cast<std::uint8_t>(nibble(text[i * 2]) * 16 + nibble(text[i * 2 + 1]));
  return out;
}
bool RepoPart(std::string_view text) {
  const auto alpha = [](char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
  };
  return !text.empty() && text.size() <= 96 && alpha(text.front()) &&
         std::ranges::all_of(text,
                             [&](char c) { return alpha(c) || c == '.' || c == '_' || c == '-'; });
}
bool Repo(std::string_view text) {
  const auto slash = text.find('/');
  return slash != std::string_view::npos && text.find("..") == std::string_view::npos &&
         RepoPart(text.substr(0, slash)) && RepoPart(text.substr(slash + 1));
}
bool Path(std::string_view text) {
  if (text.empty() || text.front() == '/' || text == "llmp-archive.json" ||
      text.find(".llmp-partial.") != text.npos || text.find(".llmp-tmp.") != text.npos ||
      !NoControlCharacters(text) || text.find_first_of(":*?\"<>|\\") != text.npos)
    return false;
  std::size_t at = 0;
  while (at < text.size()) {
    const auto end = text.find('/', at);
    const auto part = text.substr(at, end == text.npos ? text.size() - at : end - at);
    if (part.empty() || part == "." || part == ".." || part.size() > 255 || part.back() == '.' ||
        part.back() == ' ')
      return false;
    if (end == text.npos) return true;
    at = end + 1;
  }
  return false;
}
bool Text(json::Value value, std::string_view key, std::string_view expected) {
  const auto found = value.find(key);
  return found && found->is_string() && found->string() == expected;
}
bool Number(json::Value value, std::string_view key, std::int64_t expected) {
  const auto found = value.find(key);
  return found && found->int64() == expected;
}
std::optional<std::string_view> String(json::Value value, std::string_view key) {
  auto v = value.find(key);
  return v && v->is_string() ? std::optional(v->string()) : std::nullopt;
}
bool Finite(json::Value v) {
  if (v.is_number()) return v.float64() && std::isfinite(*v.float64());
  if (v.is_array() || v.is_object())
    for (std::size_t i = 0; i < v.size(); ++i)
      if (!Finite(v.is_array() ? v.at(i) : v.member(i))) return false;
  return true;
}
std::expected<json::Document, Error> ReadJson(const fs::path& path, std::uint64_t cap) {
  auto input = OpenCheckpointFile(path, kNoItem);
  if (!input) return std::unexpected(input.error());
  auto text = ReadDocument(*input, cap, kNoItem);
  if (!text) return std::unexpected(text.error());
  auto parsed = json::Parse(*text, {.max_bytes = static_cast<std::size_t>(cap),
                                    .max_depth = 32,
                                    .max_values = 1U << 21U,
                                    .max_string_bytes = 64U << 20U});
  if (!parsed || !Finite(parsed->root()))
    return Bad("archive JSON is invalid or exceeds its bounds");
  return std::move(*parsed);
}
std::expected<ArchiveInput, Error> ParseManifest(const fs::path& directory) {
  auto document = ReadJson(directory / "llmp-archive.json", kManifestCap);
  if (!document) return std::unexpected(document.error());
  auto v = document->root();
  if (!v.is_object() || !Text(v, "format", "llmp-archive-checkpoint") ||
      !Number(v, "format_version", 1) || !Text(v, "origin", "hf"))
    return Bad("unsupported archive manifest format or origin");
  auto repo = String(v, "repo"), commit = String(v, "commit");
  auto hub = v.find("hub"), files = v.find("files"), variants = v.find("variants");
  if (!repo || !Repo(*repo) || !commit || !Hex(*commit, 40) || !hub || !hub->is_object() ||
      !files || !files->is_array() || !variants || !variants->is_array())
    return Bad("invalid archive manifest fields");
  ArchiveInput out{
      .revision = directory, .repo = std::string(*repo), .commit = std::string(*commit)};
  std::set<std::string> names;
  std::string previous;
  for (std::size_t i = 0; i < files->size(); ++i) {
    auto file = files->at(i);
    if (!file.is_object()) return Bad("archive file entry is not an object");
    auto path = String(file, "path");
    auto size = file.find("size"), present = file.find("present"), lfs = file.find("lfs");
    auto sha = file.find("sha256"), oid = file.find("git_oid");
    if (!path || !Path(*path) || (!previous.empty() && previous >= *path) || !size ||
        !size->int64() || *size->int64() < 0 || !present || !present->is_bool() || !lfs ||
        !lfs->is_bool() || !sha ||
        (!sha->is_null() && (!sha->is_string() || !Hex(sha->string(), 64))) ||
        (present->boolean() && sha->is_null()) || !oid ||
        (!oid->is_null() && (!oid->is_string() || !Hex(oid->string(), 40))))
      return Bad("invalid archive file identity");
    auto lower = chat::PythonCase(*path, chat::CaseOp::kLower);
    if (!names.insert(std::move(lower)).second) return Bad("archive paths collide ignoring case");
    previous = *path;
    out.files.emplace(
        previous,
        ArchivedFile{static_cast<std::uint64_t>(*size->int64()), present->boolean(),
                     sha->is_null() ? std::nullopt : std::optional(Digest(sha->string()))});
  }
  std::set<std::string> variant_names;
  for (std::size_t i = 0; i < variants->size(); ++i) {
    auto variant = variants->at(i);
    auto id = variant.is_object() ? String(variant, "id") : std::nullopt;
    auto members = variant.is_object() ? variant.find("files") : std::nullopt;
    if (!id || id->empty() || !NoControlCharacters(*id) ||
        !variant_names.insert(std::string(*id)).second || !members || !members->is_array())
      return Bad("invalid archive variant");
    ArchiveVariant item{.id = std::string(*id)};
    std::set<std::string_view> member_names;
    for (std::size_t j = 0; j < members->size(); ++j) {
      auto name = members->at(j);
      if (!name.is_string() || !out.files.contains(name.string()) ||
          !member_names.insert(name.string()).second)
        return Bad("archive variant names an unlisted or duplicate file");
      item.files.emplace_back(name.string());
    }
    out.variants.push_back(std::move(item));
  }
  return out;
}
std::expected<bool, Error> Marker(const fs::path& root) {
  auto input = OpenCheckpointFileIfPresent(root / "llmp-archive-root.json", kNoItem);
  if (!input) return std::unexpected(input.error());
  if (!*input) return false;
  auto doc = ReadJson(root / "llmp-archive-root.json", 4096);
  if (!doc || !doc->root().is_object() || !Text(doc->root(), "format", "llmp-archive-root") ||
      !Number(doc->root(), "format_version", 1))
    return Bad("invalid archive root marker");
  return true;
}
std::expected<std::string, Error> Relative(const ArchiveInput& archive, const SourceFile& file) {
  std::error_code error;
  const auto resolved = fs::canonical(file.path, error);
  if (error) return Bad("cannot resolve archived source", Rule::kIo);
  const auto relative = resolved.lexically_relative(archive.revision).generic_string();
  if (!Path(relative)) return Bad("an archived source escapes its revision", Rule::kEscape);
  const auto lexical = fs::absolute(file.path, error)
                           .lexically_normal()
                           .lexically_relative(archive.revision)
                           .generic_string();
  if (error || !Path(lexical) || resolved != archive.revision / lexical)
    return Bad("an archived source does not resolve to its listed path", Rule::kEscape);
  return lexical;
}
// The directory of a relative archive path ("" at the revision's root).
std::string_view Directory(std::string_view path) {
  const auto slash = path.rfind('/');
  return slash == std::string_view::npos ? std::string_view() : path.substr(0, slash);
}
// Each safetensors index the import read, by its directory: the shards its
// weight_map names, revision-relative.
using Indexed = std::map<std::string, std::set<std::string>, std::less<>>;
// The variant the import's weight files (`weights`, revision-relative)
// make whole: its weight members are its `.gguf` and `.safetensors` files,
// except that a safetensors directory whose index the import read has as
// weights the shards that index names: other `.safetensors` files beside
// it (ModelOpt's amax calibration files) are not the checkpoint's weights
// and are never read (docs/model-archive.md).
std::expected<std::string, Error> SelectedVariant(const ArchiveInput& archive,
                                                  const std::set<std::string>& weights,
                                                  const Indexed& indexed = {}) {
  if (weights.empty()) return Bad("import source has no archived weights");
  std::optional<std::string> selected;
  for (const auto& variant : archive.variants) {
    std::set<std::string> members;
    bool intersects = false;
    for (const auto& name : variant.files) {
      if (!name.ends_with(".gguf") && !name.ends_with(".safetensors")) continue;
      intersects |= weights.contains(name);
      const auto index = indexed.find(Directory(name));
      if (name.ends_with(".safetensors") && index != indexed.end() && !index->second.contains(name))
        continue;
      members.insert(name);
    }
    if (!intersects) continue;
    if (members != weights || selected)
      return Bad("import source does not identify exactly one complete archive variant");
    selected = variant.id;
  }
  if (!selected) return Bad("import source belongs to no archive variant");
  return std::move(*selected);
}
// The safetensors indexes among `files` (index documents are bounded as the
// reader bounds them) with the shards their weight_maps name.
std::expected<void, Error> AddIndexes(const ArchiveInput& archive,
                                      std::span<const SourceFile> files, Indexed& out) {
  for (const auto& file : files) {
    if (!file.name.ends_with(".safetensors.index.json")) continue;
    auto path = Relative(archive, file);
    if (!path) return std::unexpected(path.error());
    auto document = ReadJson(file.path, kIndexCap);
    if (!document) return std::unexpected(document.error());
    const auto map =
        document->root().is_object() ? document->root().find("weight_map") : std::nullopt;
    if (!map || !map->is_object()) return Bad("a safetensors index has no weight map");
    const std::string directory(Directory(*path));
    auto& shards = out[directory];
    for (std::size_t i = 0; i < map->size(); ++i) {
      const auto shard = map->member(i);
      if (!shard.is_string() || !Path(shard.string()))
        return Bad("a safetensors index names an invalid shard");
      shards.insert(directory.empty() ? std::string(shard.string())
                                      : directory + '/' + std::string(shard.string()));
    }
  }
  return {};
}
}  // namespace

std::expected<std::optional<ArchiveInput>, Error> FindArchiveInput(const fs::path& source) {
  std::error_code error;
  auto absolute = fs::absolute(source, error).lexically_normal();
  if (error) return Bad("cannot resolve import path", Rule::kIo);
  fs::path at = fs::is_directory(absolute, error) ? absolute : absolute.parent_path();
  if (error) return Bad("cannot inspect import path", Rule::kIo);
  for (unsigned depth = 0; depth < 128; ++depth) {
    auto input = OpenCheckpointFileIfPresent(at / "llmp-archive.json", kNoItem);
    if (!input) return std::unexpected(input.error());
    if (*input) {
      auto resolved = fs::canonical(at, error);
      if (error) return Bad("cannot resolve archive revision", Rule::kIo);
      auto archive = ParseManifest(resolved);
      if (!archive) return std::unexpected(archive.error());
      // Only the defined archive layout can nominate a catalog destination.
      auto candidate = resolved.parent_path().parent_path().parent_path();
      auto marker = Marker(candidate);
      if (!marker) return std::unexpected(marker.error());
      if (*marker) {
        const auto slash = archive->repo.find('/');
        const auto expected = candidate / "hf" / archive->repo.substr(0, slash) /
                              (archive->repo.substr(slash + 1) + "@" + archive->commit);
        if (resolved != expected)
          return Bad("revision location disagrees with its archive identity");
        archive->root = candidate;
      }
      return std::optional<ArchiveInput>(std::move(*archive));
    }
    auto marker = Marker(at);
    if (!marker) return std::unexpected(marker.error());
    if (*marker) return Bad("source is inside an archive but has no revision manifest");
    if (at == at.parent_path()) return std::optional<ArchiveInput>();
    at = at.parent_path();
  }
  return Bad("archive ancestor search exceeds its bound", Rule::kBounds);
}

std::expected<fs::path, Error> SelectArchiveSource(const ArchiveInput& archive,
                                                   const fs::path& requested) {
  std::error_code error;
  const auto path = fs::absolute(requested, error).lexically_normal();
  if (error) return Bad("cannot resolve selected source", Rule::kIo);
  if (!fs::is_directory(path, error)) return path;
  if (error) return Bad("cannot examine selected source", Rule::kIo);
  const auto relative = path.lexically_relative(archive.revision).generic_string();
  const auto prefix = relative == "." ? std::string() : relative + '/';
  std::vector<const ArchiveVariant*> candidates;
  for (const auto& variant : archive.variants) {
    if (std::ranges::any_of(variant.files, [&](const auto& name) {
          return name.starts_with(prefix) &&
                 (name.ends_with(".gguf") || name.ends_with(".safetensors"));
        }))
      candidates.push_back(&variant);
  }
  if (candidates.size() != 1)
    return Bad("selected archive directory has zero or multiple variants");
  std::vector<std::string_view> weights;
  bool safetensors = false;
  for (const auto& name : candidates.front()->files)
    if (name.ends_with(".gguf"))
      weights.push_back(name);
    else if (name.ends_with(".safetensors"))
      safetensors = true;
  if (!weights.empty() && safetensors)
    return Bad("selected archive variant mixes independent weight containers");
  if (!weights.empty()) {
    std::ranges::sort(weights);
    return archive.revision / weights.front();
  }
  return path;
}

std::expected<std::vector<std::optional<base::Sha256Digest>>, Error> ArchiveFileIdentities(
    const ArchiveInput& archive, std::span<const SourceFile> files) {
  std::vector<std::optional<base::Sha256Digest>> expected;
  expected.reserve(files.size());
  for (const auto& file : files) {
    auto path = Relative(archive, file);
    if (!path) return std::unexpected(path.error());
    const auto entry = archive.files.find(*path);
    if (entry == archive.files.end() || !entry->second.present || !entry->second.sha256 ||
        entry->second.bytes != file.bytes)
      return Bad("a source file is not present with matching size in the archive manifest");
    expected.push_back(entry->second.sha256);
  }
  return expected;
}

std::expected<std::vector<std::optional<base::Sha256Digest>>, Error> ArchiveIdentities(
    const ArchiveInput& archive, const Checkpoint& source) {
  auto expected = ArchiveFileIdentities(archive, source.files);
  if (!expected) return expected;
  std::set<std::string> weights;
  for (const auto& file : source.files) {
    auto path = Relative(archive, file);
    if (path && (path->ends_with(".gguf") || path->ends_with(".safetensors"))) {
      weights.insert(std::move(*path));
    }
  }
  if (source.container == "diffusers") {
    for (const auto& comp : source.components) {
      for (const auto& file : comp.files) {
        auto path = Relative(archive, file);
        if (!path) return std::unexpected(path.error());
        const auto entry = archive.files.find(*path);
        if (entry == archive.files.end() || !entry->second.present || !entry->second.sha256 ||
            entry->second.bytes != file.bytes)
          return Bad("a source file is not present with matching size in the archive manifest");
        if (path->ends_with(".safetensors")) weights.insert(*path);
      }
    }
  }
  Indexed indexes;
  if (auto added = AddIndexes(archive, source.files, indexes); !added)
    return std::unexpected(added.error());
  if (source.container == "diffusers")
    for (const auto& comp : source.components)
      if (auto added = AddIndexes(archive, comp.files, indexes); !added)
        return std::unexpected(added.error());
  if (auto selected = SelectedVariant(archive, weights, indexes); !selected)
    return std::unexpected(selected.error());
  return expected;
}

namespace {
std::expected<void, Error> WriteArchiveEvent(const ArchiveInput& archive,
                                             std::string_view format_name,
                                             const std::string& artifact_id,
                                             const artifact::Converter& importer,
                                             std::string_view variant) {
  std::array<char, 256> hostname{};
  if (::gethostname(hostname.data(), hostname.size() - 1) != 0)
    return Bad("cannot name import host", Rule::kIo);
  std::string host;
  for (char c : std::string_view(hostname.data())) {
    if (c == '.' || host.size() == 32) break;
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    host += ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-') ? c : '-';
  }
  if (host.empty()) host = "host";
  std::array<std::byte, 16> random{};
  if (!platform::FillRandom(random))
    return Bad("cannot generate catalog event identity", Rule::kIo);
  const auto now = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
  const auto time = std::format("{:%FT%TZ}", now);
  const auto id = std::format("{:%Y%m%dT%H%M%SZ}-{}-{}-{}", now, host, ::getpid(),
                              base::ToHex(base::Sha256().Update(random).Finish()).substr(0, 16));
  const auto importer_json = std::format(
      "{{\"build\":{},\"name\":{},\"options_sha256\":{},\"version\":{}}}", Q(importer.build),
      Q(importer.name), Q(base::ToHex(importer.options_sha256)), Q(importer.version));
  const auto text = std::format(
      "{{\"body\":{{\"artifact\":{{\"format\":{},\"format_version\":1,\"id\":{},"
      "\"importer\":{}}},\"commit\":{},\"host\":{},\"repo\":{},\"variant\":{}}},\"format\":\"llmp-"
      "archive-event\",\"format_version\":1,\"host\":{},\"id\":{},\"kind\":\"import\",\"time\":{},"
      "\"tool\":\"llmp-import/1\"}}\n",
      Q(format_name), Q(artifact_id), importer_json, Q(archive.commit), Q(host), Q(archive.repo),
      Q(variant), Q(host), Q(id), Q(time));
  auto marker = Marker(archive.root);
  if (!marker || !*marker)
    return Bad("archive root disappeared before event publication", Rule::kIo);
  std::error_code error;
  // Never create the root (including after an unmount).
  for (const auto& directory : {archive.root / "catalog", archive.root / "catalog/events"}) {
    if (auto ok = CreateImportDirectory(directory, false); !ok) return ok;
  }
  const auto events = archive.root / "catalog/events";
  const auto temporary = events / (id + ".json.llmp-tmp.native");
  artifact::FileDescriptor file(
      ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600));
  if (!file.valid()) return Bad("cannot create archive event", Rule::kIo);
  const auto abandon = [&]() {
    std::error_code ignored;
    fs::remove(temporary, ignored);
  };
  std::size_t written = 0;
  while (written < text.size()) {
    const auto n = ::write(file.get(), text.data() + written, text.size() - written);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) {
      abandon();
      return Bad("cannot write archive event", Rule::kIo);
    }
    written += static_cast<std::size_t>(n);
  }
  if (::fsync(file.get()) != 0) {
    abandon();
    return Bad("cannot sync archive event", Rule::kIo);
  }
  marker = Marker(archive.root);
  if (!marker || !*marker) {
    abandon();
    return Bad("archive root disappeared before event rename", Rule::kIo);
  }
  fs::rename(temporary, events / (id + ".json"), error);
  if (error) {
    abandon();
    return Bad("cannot publish archive event", Rule::kIo);
  }
  artifact::FileDescriptor directory(::open(events.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  if (!directory.valid() || ::fsync(directory.get()) != 0)
    return Bad("cannot sync archive event directory", Rule::kIo);
  return {};
}
}  // namespace

std::expected<void, Error> RecordArchiveImport(const ArchiveInput& archive,
                                               const Checkpoint& source,
                                               const artifact::Artifact& artifact) {
  if (archive.root.empty()) return {};
  if (artifact.sources().size() != source.files.size())
    return Bad("published artifact source list disagrees with the archive input");
  std::set<std::string> source_names;
  for (const auto& file : source.files) {
    auto relative = Relative(archive, file);
    if (!relative) return std::unexpected(relative.error());
    const auto expected = archive.files.find(*relative);
    const auto actual = std::ranges::find(artifact.sources(), file.name, &artifact::Source::name);
    if (expected == archive.files.end() || !expected->second.present || !expected->second.sha256 ||
        actual == artifact.sources().end() || actual->bytes.value() != expected->second.bytes ||
        actual->sha256 != *expected->second.sha256)
      return Bad("published artifact source identity disagrees with the archive manifest");
    source_names.insert(std::move(*relative));
  }
  std::erase_if(source_names, [](const auto& name) {
    return !name.ends_with(".gguf") && !name.ends_with(".safetensors");
  });
  Indexed indexes;
  if (auto added = AddIndexes(archive, source.files, indexes); !added)
    return std::unexpected(added.error());
  auto variant = SelectedVariant(archive, source_names, indexes);
  if (!variant) return std::unexpected(variant.error());
  return WriteArchiveEvent(archive, "llmp-artifact", artifact.id(), artifact.converter(), *variant);
}

std::expected<void, Error> RecordArchiveComponentImport(const ArchiveInput& archive,
                                                        const Checkpoint& source,
                                                        const artifact::Artifact& artifact,
                                                        std::optional<std::string_view> caller) {
  if (archive.root.empty()) return {};
  std::set<std::string> weights;
  for (const auto& file : source.files) {
    auto relative = Relative(archive, file);
    if (!relative) return std::unexpected(relative.error());
    if (relative->ends_with(".gguf") || relative->ends_with(".safetensors"))
      weights.insert(std::move(*relative));
  }
  std::size_t outside = 0;
  for (const auto& actual : artifact.sources()) {
    const auto file = std::ranges::find(source.files, actual.name, &SourceFile::name);
    if (file == source.files.end()) {
      if (!caller || actual.name != *caller || ++outside > 1)
        return Bad("published component has sources outside its checkpoint");
      continue;
    }
    auto relative = Relative(archive, *file);
    if (!relative) return std::unexpected(relative.error());
    const auto expected = archive.files.find(*relative);
    if (expected == archive.files.end() || !expected->second.present || !expected->second.sha256 ||
        actual.bytes.value() != expected->second.bytes || actual.sha256 != *expected->second.sha256)
      return Bad("published component source identity disagrees with the archive manifest");
  }
  Indexed indexes;
  if (auto added = AddIndexes(archive, source.files, indexes); !added)
    return std::unexpected(added.error());
  auto variant = SelectedVariant(archive, weights, indexes);
  if (!variant) return std::unexpected(variant.error());
  return WriteArchiveEvent(archive, "llmp-artifact", artifact.id(), artifact.converter(), *variant);
}

std::expected<void, Error> RecordArchiveImport(const ArchiveInput& archive,
                                               const Checkpoint& source,
                                               const artifact::Composition& composition) {
  if (archive.root.empty()) return {};
  if (composition.sources().size() != source.files.size())
    return Bad("published composition source list disagrees with the archive input");
  for (const auto& file : source.files) {
    auto relative = Relative(archive, file);
    if (!relative) return std::unexpected(relative.error());
    const auto expected = archive.files.find(*relative);
    const std::string name = fs::path(file.name).filename().string();
    const auto actual = std::ranges::find(composition.sources(), name, &artifact::Source::name);
    if (expected == archive.files.end() || !expected->second.present || !expected->second.sha256 ||
        actual == composition.sources().end() || actual->bytes.value() != expected->second.bytes ||
        actual->sha256 != *expected->second.sha256)
      return Bad("published composition source identity disagrees with the archive manifest");
  }
  std::set<std::string> source_names;
  Indexed indexes;
  for (const auto& comp : source.components) {
    for (const auto& file : comp.files) {
      auto relative = Relative(archive, file);
      if (!relative) return std::unexpected(relative.error());
      if (relative->ends_with(".safetensors")) {
        source_names.insert(std::move(*relative));
      }
    }
    if (auto added = AddIndexes(archive, comp.files, indexes); !added)
      return std::unexpected(added.error());
  }
  auto variant = SelectedVariant(archive, source_names, indexes);
  if (!variant) return std::unexpected(variant.error());
  return WriteArchiveEvent(archive, "llmp-composition", composition.id(), composition.converter(),
                           *variant);
}
}  // namespace llmp::import
