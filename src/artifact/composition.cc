// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The checks follow docs/experiments/artifact-layout/import_m3.py's
// verify_composition, in its order, so that a malformed composition fails
// the same rule here as in the prototype (the oracle).

#include "artifact/composition.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "artifact/error.h"
#include "artifact/json.h"
#include "artifact/schema.h"
#include "base/bytes.h"
#include "base/sha256.h"

namespace llmp::artifact {

const CompositionComponent* Composition::Find(std::string_view role) const {
  const auto it = std::ranges::find(components_, role, &CompositionComponent::role);
  return it == components_.end() ? nullptr : &*it;
}

std::optional<std::string_view> Composition::Metadata(std::string_view name) const {
  const auto it = std::ranges::find(kept_, name, &KeptFile::name);
  if (it == kept_.end()) {
    return std::nullopt;
  }
  return std::string_view(it->bytes);
}

namespace {

using schema::Fail;
using Step = std::expected<void, Error>;

constexpr std::string_view kFormat = "jitllm-composition";
constexpr std::size_t kMaxEntriesPerDirectory = 1024;

template <typename T>
std::unexpected<Error> Pass(const std::expected<T, Error>& failed) {
  return std::unexpected(failed.error());
}

// [a-z][a-z0-9_]{0,63}: a component's role.
constexpr bool IsRole(std::string_view s) {
  if (s.empty() || s.size() > 64 || s[0] < 'a' || s[0] > 'z') {
    return false;
  }
  return std::ranges::all_of(
      s, [](char c) { return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_'; });
}

// The names in `dir` (a fresh descriptor of it, so `dir` keeps its
// position), refusing links and anything but regular, singly linked files
// and, where `dirs` is given, directories.
std::expected<std::vector<std::string>, Error> Scan(int dir, std::vector<std::string>* dirs) {
  const int fresh = ::openat(dir, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (fresh < 0) {
    return Fail(Rule::kFileType, "unreadable directory");
  }
  DIR* stream = ::fdopendir(fresh);
  if (stream == nullptr) {
    ::close(fresh);
    return Fail(Rule::kFileType, "unreadable directory");
  }
  std::vector<std::string> files;
  std::expected<std::vector<std::string>, Error> result;
  std::size_t seen = 0;
  while (true) {
    errno = 0;
    const dirent* entry = ::readdir(stream);  // NOLINT(concurrency-mt-unsafe): own stream
    if (entry == nullptr) {
      if (errno != 0) {
        result = Fail(Rule::kFileType, "unreadable directory");
      }
      break;
    }
    const std::string name = entry->d_name;
    if (name == "." || name == "..") {
      continue;
    }
    if (++seen > kMaxEntriesPerDirectory) {
      result = Fail(Rule::kFileSet, "more directory entries than a composition lists");
      break;
    }
    struct stat status{};
    if (::fstatat(dir, name.c_str(), &status, AT_SYMLINK_NOFOLLOW) != 0) {
      result = Fail(Rule::kFileType, "unreadable directory entry");
      break;
    }
    if (S_ISLNK(status.st_mode)) {
      result = Fail(Rule::kFileType, "a symbolic link in the composition");
      break;
    }
    if (S_ISDIR(status.st_mode) && dirs != nullptr) {
      dirs->push_back(name);
      continue;
    }
    if (!S_ISREG(status.st_mode) || status.st_nlink != 1) {
      result = Fail(S_ISDIR(status.st_mode) ? Rule::kFileSet : Rule::kFileType,
                    S_ISDIR(status.st_mode) ? "an unexpected directory"
                                            : "not a regular, singly linked file");
      break;
    }
    files.push_back(name);
  }
  ::closedir(stream);
  if (!result) {
    return result;
  }
  std::ranges::sort(files);
  return files;
}

// A whole regular file of at most `limit` bytes, and `size` when given.
std::expected<std::string, Error> ReadFile(int dir, const char* name,
                                           std::optional<std::uint64_t> size, std::uint64_t limit,
                                           std::uint64_t item) {
  if (size && *size > limit) {
    return Fail(Rule::kFileSize, "document larger than its cap", item);
  }
  const int fd = ::openat(dir, name, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC | O_NOCTTY);
  if (fd < 0) {
    return Fail(Rule::kFileType, "cannot open a listed file as a regular file", item);
  }
  const FileDescriptor owned(fd);
  struct stat status{};
  if (::fstat(fd, &status) != 0) {
    return Fail(Rule::kIo, "fstat failed", item);
  }
  if (!S_ISREG(status.st_mode) || status.st_nlink != 1) {
    return Fail(Rule::kFileType, "not a regular, singly linked file", item);
  }
  if (size && std::cmp_not_equal(status.st_size, *size)) {
    return Fail(Rule::kFileSize, "size differs from the manifest", item);
  }
  std::string data;
  while (true) {
    const std::size_t have = data.size();
    const std::size_t want = std::min<std::uint64_t>(limit + 1 - have, std::uint64_t{1} << 20U);
    data.resize(have + want);
    ssize_t n = 0;
    do {
      n = ::pread(fd, data.data() + have, want, static_cast<off_t>(have));
    } while (n < 0 && errno == EINTR);
    if (n < 0) {
      return Fail(Rule::kIo, "read failed", item);
    }
    data.resize(have + static_cast<std::size_t>(n));
    if (data.size() > limit) {
      return Fail(Rule::kFileSize, "document larger than its cap", item);
    }
    if (std::cmp_less(n, want)) {
      return data;
    }
  }
}

}  // namespace

class CompositionOpener {
 public:
  CompositionOpener(std::filesystem::path root, std::optional<std::string> expected_id)
      : root_path_(std::move(root)), expected_id_(std::move(expected_id)) {}

  std::expected<Composition, Error> Run() {
    for (auto step : {&CompositionOpener::Walk, &CompositionOpener::ReadManifest,
                      &CompositionOpener::CheckManifest, &CompositionOpener::CheckFiles}) {
      if (auto ok = (this->*step)(); !ok) {
        return std::unexpected(ok.error());
      }
    }
    return std::move(out_);
  }

 private:
  Step Walk() {
    std::filesystem::path root = root_path_;
    while (root.has_relative_path() && !root.has_filename()) {
      root = root.parent_path();  // "a/b/" names b
    }
    out_.id_ = expected_id_ ? *expected_id_ : root.filename().string();
    root_ = FileDescriptor(::open(root.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
    if (!root_.valid()) {
      return Fail(Rule::kFileType, "the composition root must be a real directory");
    }
    std::vector<std::string> dirs;
    auto files = Scan(root_.get(), &dirs);
    if (!files) {
      return Pass(files);
    }
    for (const std::string& name : *files) {
      present_.push_back(name);
    }
    for (const std::string& d : dirs) {
      if (d != "data" && d != "meta") {
        return Fail(Rule::kFileSet, "an unexpected directory");
      }
      has_data_ = has_data_ || d == "data";
      FileDescriptor fd(
          ::openat(root_.get(), d.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
      if (!fd.valid()) {
        return Fail(Rule::kFileType, "unreadable directory");
      }
      auto nested = Scan(fd.get(), nullptr);
      if (!nested) {
        return Pass(nested);
      }
      for (const std::string& name : *nested) {
        std::string path = d;
        path += '/';
        path += name;
        present_.push_back(std::move(path));
      }
      if (d == "meta") {
        meta_ = std::move(fd);
      }
    }
    std::ranges::sort(present_);
    return {};
  }

  Step ReadManifest() {
    auto bytes = ReadFile(root_.get(), "manifest.json", std::nullopt, kMaxManifestBytes, kNoItem);
    if (!bytes) {
      return Pass(bytes);
    }
    bytes_ = std::move(*bytes);
    auto parsed = json::Parse(bytes_);
    if (!parsed) {
      return Pass(parsed);
    }
    doc_ = std::move(*parsed);
    const json::Value m = doc_->root();
    const auto format = m.find("format");
    if (!m.is_object() || !format || !schema::IsString(*format, kFormat)) {
      return Fail(Rule::kFormat, "not a llmpalooza composition");
    }
    // import_m3.py checks for data after the format, as here.
    if (has_data_) {
      return Fail(Rule::kFileSet, "a composition holds no data shards");
    }
    const auto version = m.find("format_version");
    if (!version || !schema::IsInt(*version, 0)) {
      return Fail(Rule::kUnsupportedVersion, "format_version is not 0");
    }
    if (!doc_->canonical()) {
      return Fail(Rule::kCanonical, "manifest.json is not in canonical form");
    }
    if (out_.id_ != base::ToHex(base::Sha256().Update(bytes_).Finish())) {
      return Fail(Rule::kIdentity, "the composition name is not the manifest digest");
    }
    return {};
  }

  Step CheckManifest() {
    if (!doc_) {
      return Fail(Rule::kFormat, "no manifest was read");
    }
    const json::Value m = doc_->root();
    if (auto ok = schema::Object(m,
                                 {"format", "format_version", "experimental", "model", "components",
                                  "source", "converter", "files"},
                                 {}, "composition keys");
        !ok) {
      return ok;
    }
    if (schema::Get(m, "experimental").kind() != json::Kind::kTrue) {
      return Fail(Rule::kSchema, "experimental must be true before D-018's gate");
    }
    const json::Value model = schema::Get(m, "model");
    if (auto ok = schema::Object(model, {"architecture"}, {}, "model keys"); !ok) {
      return ok;
    }
    auto arch =
        schema::Str(schema::Get(model, "architecture"), schema::IsName, "model.architecture");
    if (!arch) {
      return Pass(arch);
    }
    out_.architecture_ = std::string(*arch);
    const json::Value components = schema::Get(m, "components");
    if (auto ok = schema::List(components, 1, "components"); !ok) {
      return ok;
    }
    if (components.size() > kMaxComponents) {
      return Fail(Rule::kSchema, "more than 16 components");
    }
    for (std::size_t i = 0; i < components.size(); ++i) {
      const json::Value c = components.at(i);
      if (auto ok =
              schema::Object(c, {"role", "artifact", "architecture"}, {}, "component keys", i);
          !ok) {
        return ok;
      }
      auto role = schema::Str(schema::Get(c, "role"), IsRole, "component.role", i);
      if (!role) {
        return Pass(role);
      }
      auto id = schema::Str(schema::Get(c, "artifact"), schema::IsHex, "component.artifact", i);
      if (!id) {
        return Pass(id);
      }
      auto carch =
          schema::Str(schema::Get(c, "architecture"), schema::IsName, "component.architecture", i);
      if (!carch) {
        return Pass(carch);
      }
      if (!out_.components_.empty() && out_.components_.back().role >= *role) {
        return Fail(Rule::kCanonical, "components not in canonical order or duplicated", i);
      }
      out_.components_.push_back({.role = std::string(*role),
                                  .artifact = std::string(*id),
                                  .architecture = std::string(*carch)});
    }
    const json::Value sources = schema::Get(m, "source");
    if (auto ok = schema::List(sources, 1, "source"); !ok) {
      return ok;
    }
    for (std::size_t i = 0; i < sources.size(); ++i) {
      const json::Value s = sources.at(i);
      if (auto ok = schema::Object(s, {"name", "bytes", "sha256"}, {}, "source keys", i); !ok) {
        return ok;
      }
      auto name = schema::Str(schema::Get(s, "name"), schema::IsName, "source.name", i);
      if (!name) {
        return Pass(name);
      }
      auto bytes = schema::Int(schema::Get(s, "bytes"), 0, schema::kMaxInt, "source.bytes", i);
      if (!bytes) {
        return Pass(bytes);
      }
      auto sha = schema::Str(schema::Get(s, "sha256"), schema::IsHex, "source.sha256", i);
      if (!sha) {
        return Pass(sha);
      }
      if (!out_.sources_.empty() && out_.sources_.back().name >= *name) {
        return Fail(Rule::kCanonical, "source not in canonical order or duplicated", i);
      }
      out_.sources_.push_back({.name = std::string(*name),
                               .bytes = Bytes(*bytes),
                               .sha256 = schema::DigestFromHex(*sha).value_or(Digest{})});
    }
    const json::Value conv = schema::Get(m, "converter");
    if (auto ok = schema::Object(conv, {"name", "version"}, {}, "converter keys"); !ok) {
      return ok;
    }
    auto name = schema::Str(schema::Get(conv, "name"), schema::IsText, "converter.name");
    if (!name) {
      return Pass(name);
    }
    auto version = schema::Str(schema::Get(conv, "version"), schema::IsText, "converter.version");
    if (!version) {
      return Pass(version);
    }
    out_.converter_ = {.name = std::string(*name), .version = std::string(*version)};
    const json::Value files = schema::Get(m, "files");
    if (auto ok = schema::List(files, 1, "files"); !ok) {
      return ok;
    }
    for (std::size_t i = 0; i < files.size(); ++i) {
      const json::Value f = files.at(i);
      if (auto ok = schema::Object(f, {"path", "role", "bytes", "sha256"}, {}, "file keys", i);
          !ok) {
        return ok;
      }
      auto path = schema::Str(schema::Get(f, "path"), schema::IsMetaPath, "file.path", i);
      if (!path) {
        return Pass(path);
      }
      if (!schema::IsString(schema::Get(f, "role"), "source-metadata")) {
        return Fail(Rule::kPath, "a composition lists kept metadata only", i);
      }
      auto bytes = schema::Int(schema::Get(f, "bytes"), 0,
                               static_cast<std::int64_t>(kMaxMetadataBytes), "file.bytes", i);
      if (!bytes) {
        return Pass(bytes);
      }
      auto sha = schema::Str(schema::Get(f, "sha256"), schema::IsHex, "file.sha256", i);
      if (!sha) {
        return Pass(sha);
      }
      if (!listed_.empty() && listed_.back().path >= *path) {
        return Fail(Rule::kCanonical, "files not in canonical order or duplicated", i);
      }
      listed_.push_back({.path = std::string(*path),
                         .role = FileRole::kSourceMetadata,
                         .bytes = Bytes(*bytes),
                         .sha256 = schema::DigestFromHex(*sha).value_or(Digest{})});
    }
    return {};
  }

  Step CheckFiles() {
    if (std::ranges::find(listed_, std::string("meta/model_index.json"), &ListedFile::path) ==
        listed_.end()) {
      return Fail(Rule::kFileSet, "a composition keeps its model_index.json");
    }
    std::vector<std::string> expected = {"manifest.json"};
    for (const ListedFile& f : listed_) {
      expected.push_back(f.path);
    }
    std::ranges::sort(expected);
    if (expected != present_) {
      return Fail(Rule::kFileSet, "files differ from the manifest's list");
    }
    if (out_.sources_.size() != listed_.size()) {
      return Fail(Rule::kFileSet, "every source is a kept file and every kept file a source");
    }
    std::uint64_t total = 0;
    for (std::size_t i = 0; i < listed_.size(); ++i) {
      const ListedFile& f = listed_[i];
      const std::string name = f.path.substr(5);
      // Both lists are sorted, and "meta/" is a common prefix.
      const Source& s = out_.sources_[i];
      if (s.name != name) {
        return Fail(Rule::kFileSet, "every source is a kept file and every kept file a source", i);
      }
      if (s.bytes != f.bytes || s.sha256 != f.sha256) {
        return Fail(Rule::kFileSet, "a kept file is not a verbatim copy of a recorded source", i);
      }
      total += f.bytes.value();
      if (total > kMaxMetadataTotal) {
        return Fail(Rule::kFileSize, "kept metadata exceeds its total cap", i);
      }
      auto data = ReadFile(meta_.get(), name.c_str(), f.bytes.value(), kMaxMetadataBytes, i);
      if (!data) {
        return Pass(data);
      }
      if (base::Sha256().Update(*data).Finish() != f.sha256) {
        return Fail(Rule::kHash, "a kept file's digest differs from the manifest", i);
      }
      out_.kept_.push_back({.name = name, .bytes = std::move(*data)});
    }
    return {};
  }

  std::filesystem::path root_path_;
  std::optional<std::string> expected_id_;
  FileDescriptor root_;
  FileDescriptor meta_;
  std::vector<std::string> present_;
  bool has_data_ = false;
  std::string bytes_;
  std::optional<json::Document> doc_;
  std::vector<ListedFile> listed_;
  Composition out_;
};

std::expected<Composition, Error> OpenComposition(const std::filesystem::path& root,
                                                  const std::optional<std::string>& expected_id) {
  return CompositionOpener(root, expected_id).Run();
}

}  // namespace llmp::artifact
