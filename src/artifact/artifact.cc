// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The checks follow docs/experiments/artifact-layout/layout.py's _verify
// with deep=False, in its order, so that a malformed artifact fails the same
// rule here as in the prototype (the oracle, D-056). Comments name the
// prototype's function where the correspondence is not obvious.

#include "artifact/artifact.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <filesystem>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

#include "artifact/error.h"
#include "artifact/json.h"
#include "artifact/layout.h"
#include "artifact/representation.h"
#include "artifact/schema.h"
#include "base/bytes.h"
#include "base/check.h"
#include "base/sha256.h"
#include "platform/direct_io.h"
#include "platform/files.h"
#include "platform/path_trust.h"

namespace llmp::artifact {

FileDescriptor& FileDescriptor::operator=(FileDescriptor&& other) noexcept {
  if (this != &other) {
    if (fd_ >= 0) {
      ::close(fd_);
    }
    fd_ = std::exchange(other.fd_, -1);
  }
  return *this;
}

FileDescriptor::~FileDescriptor() {
  if (fd_ >= 0) {
    ::close(fd_);
  }
}

namespace {

using schema::Fail;
using Item = std::uint64_t;

constexpr std::string_view kManifestFormat = "jitllm-artifact";
constexpr std::string_view kIndexFormat = "jitllm-index";
constexpr std::string_view kShardPrefix =
    R"({"__metadata__":{"format":"jitllm-shard","format_version":"0"})";
constexpr std::uint64_t kSaturated = std::numeric_limits<std::uint64_t>::max();
constexpr std::size_t kReadPiece = std::size_t{1} << 20U;

std::uint64_t SatAdd(std::uint64_t a, std::uint64_t b) {
  std::uint64_t out = 0;
  return __builtin_add_overflow(a, b, &out) ? kSaturated : out;
}

// Rounded up to a multiple of `unit`, or saturated.
std::uint64_t AlignUp(std::uint64_t n, std::uint64_t unit) {
  const std::uint64_t padded = SatAdd(n, unit - 1);
  return padded == kSaturated ? kSaturated : padded / unit * unit;
}

Digest Sha256Of(std::string_view bytes) { return base::Sha256().Update(bytes).Finish(); }

std::uint64_t Chunks(std::uint64_t stored) {
  return (stored / kChunkBytes) + (stored % kChunkBytes != 0 ? 1 : 0);
}

template <typename T>
std::unexpected<Error> Pass(const std::expected<T, Error>& failed) {
  return std::unexpected(failed.error());
}

// Opens `name` beneath dir without following a link, and without blocking
// on a FIFO swapped in; kFileType if it is not a regular, singly linked
// file.
std::expected<FileDescriptor, Error> OpenRegular(int dir, const char* name, struct stat& status,
                                                 Item item) {
  FileDescriptor fd(::openat(dir, name, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC | O_NOCTTY));
  if (!fd.valid()) {
    return Fail(Rule::kFileType, "cannot open a listed file as a regular file", item);
  }
  if (::fstat(fd.get(), &status) != 0) {
    return Fail(Rule::kIo, "fstat failed", item);
  }
  if (!S_ISREG(status.st_mode) || status.st_nlink != 1) {
    return Fail(Rule::kFileType, "not a regular, singly linked file", item);
  }
  return fd;
}

// The file's inode generation (platform::FileGeneration). Any failure is
// "none"; availability that differs between open and reopen fails the
// comparison, so it never skips it.
std::optional<std::uint64_t> Generation(int fd) { return platform::FileGeneration(fd); }

// Reads length bytes at offset; fewer only at the end of the file.
std::expected<std::size_t, Error> ReadAt(int fd, std::uint64_t offset, std::span<char> out,
                                         Item item) {
  std::size_t done = 0;
  while (done < out.size()) {
    const ssize_t n =
        ::pread(fd, out.data() + done, out.size() - done, static_cast<off_t>(offset + done));
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      return Fail(Rule::kIo, "read failed", item);
    }
    if (n == 0) {
      break;
    }
    done += static_cast<std::size_t>(n);
  }
  return done;
}

// A whole document of at most `limit` bytes (layout.py _read_doc): larger
// is kFileSize, whatever the file claims.
std::expected<std::string, Error> ReadDocument(int dir, const char* name,
                                               std::optional<std::uint64_t> size,
                                               std::uint64_t limit, Item item) {
  if (size && *size > limit) {
    return Fail(Rule::kFileSize, "document larger than its cap", item);
  }
  struct stat status{};
  auto fd = OpenRegular(dir, name, status, item);
  if (!fd) {
    return Pass(fd);
  }
  if (size && std::cmp_not_equal(status.st_size, *size)) {
    return Fail(Rule::kFileSize, "size differs from the manifest", item);
  }
  std::string data;
  std::size_t have = 0;
  while (true) {
    const std::size_t want = std::min<std::uint64_t>(limit + 1 - have, kReadPiece);
    data.resize(have + want);
    auto n = ReadAt(fd->get(), have, std::span(data).subspan(have, want), item);
    if (!n) {
      return Pass(n);
    }
    have += *n;
    data.resize(have);
    if (have > limit) {
      return Fail(Rule::kFileSize, "document larger than its cap", item);
    }
    if (*n < want) {
      return data;
    }
  }
}

// The names in a directory, through a fresh open of it (so the caller's
// descriptor keeps its own position).
std::expected<std::vector<std::string>, Error> ListNames(int dir, std::size_t& budget, Item item) {
  const int fresh = ::openat(dir, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (fresh < 0) {
    return Fail(Rule::kFileType, "unreadable directory", item);
  }
  DIR* stream = ::fdopendir(fresh);
  if (stream == nullptr) {
    ::close(fresh);
    return Fail(Rule::kFileType, "unreadable directory", item);
  }
  std::vector<std::string> names;
  std::expected<std::vector<std::string>, Error> result;
  while (true) {
    errno = 0;
    const dirent* entry = ::readdir(stream);  // NOLINT(concurrency-mt-unsafe): own stream
    if (entry == nullptr) {
      if (errno != 0) {
        result = Fail(Rule::kFileType, "unreadable directory", item);
      }
      break;
    }
    const std::string_view name = entry->d_name;
    if (name == "." || name == "..") {
      continue;
    }
    if (budget == 0) {
      result = Fail(Rule::kFileSet, "more directory entries than any manifest lists", item);
      break;
    }
    --budget;
    names.emplace_back(name);
  }
  ::closedir(stream);
  if (!result) {
    return result;
  }
  std::ranges::sort(names);
  return names;
}

struct Span {
  std::uint32_t group = 0;
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
  std::uint64_t readable = 0;
  std::string name;
  ContainerView view;
};

bool SpanLess(const Span& a, const Span& b) {
  return std::tie(a.group, a.offset, a.bytes, a.readable, a.name) <
         std::tie(b.group, b.offset, b.bytes, b.readable, b.name);
}

}  // namespace

class Opener {
 public:
  Opener(std::filesystem::path root, OpenOptions options)
      : root_path_(std::move(root)), options_(std::move(options)) {}

  std::expected<Artifact, Error> Run() {
    for (auto step : {&Opener::OpenRoot, &Opener::Walk, &Opener::ReadManifest,
                      &Opener::CheckManifest, &Opener::CheckFileSet, &Opener::CheckMetadata,
                      &Opener::ReadIndex, &Opener::CheckIndex, &Opener::CheckShards}) {
      if (auto ok = (this->*step)(); !ok) {
        return std::unexpected(ok.error());
      }
    }
    Finish();
    return std::move(artifact_);
  }

 private:
  using Step = std::expected<void, Error>;

  // --- the directory (layout.py _verify's root check and _walk)

  Step OpenRoot() {
    std::filesystem::path root = root_path_;
    while (root.has_relative_path() && !root.has_filename()) {
      root = root.parent_path();  // "a/b/" names b
    }
    artifact_.id_ = options_.expected_id ? *options_.expected_id : root.filename().string();
    if (options_.trusted_owner) {
      std::error_code error;
      const std::filesystem::path absolute = std::filesystem::absolute(root, error);
      if (error) {
        return Fail(Rule::kIo, "cannot make the artifact path absolute");
      }
      auto walk = platform::WalkTrusted(absolute, *options_.trusted_owner, false);
      if (!walk) {
        return Fail(Rule::kUntrusted, "the path to the artifact is not trusted");
      }
      if (!walk->exists) {
        return Fail(Rule::kFileType, "the artifact root must be a real directory");
      }
      trusted_root_ = walk->status;
    }
    artifact_.root_ =
        FileDescriptor(::open(root.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
    if (!artifact_.root_.valid()) {
      return Fail(Rule::kFileType, "the artifact root must be a real directory");
    }
    struct stat status{};
    if (::fstat(artifact_.root_.get(), &status) != 0) {
      return Fail(Rule::kIo, "fstat failed");
    }
    if (options_.trusted_owner) {
      if (status.st_dev != trusted_root_.st_dev || status.st_ino != trusted_root_.st_ino) {
        return Fail(Rule::kUntrusted, "the artifact root changed during the trust walk");
      }
      if (auto ok = Trusted(status, root); !ok) {
        return ok;
      }
    }
    return {};
  }

  Step Trusted(const struct stat& status, const std::filesystem::path& path) const {
    if (!options_.trusted_owner) {
      return {};
    }
    const uid_t trusted = *options_.trusted_owner;
    if ((status.st_uid != 0 && status.st_uid != trusted) ||
        platform::OthersCanWrite(status, trusted, path)) {
      return Fail(Rule::kUntrusted, "a file or directory others could change");
    }
    return {};
  }

  // Every entry of one directory: links are refused, directories are
  // returned, regular singly linked files are recorded as present.
  std::expected<std::vector<std::string>, Error> Scan(int dir, std::string_view prefix,
                                                      const std::filesystem::path& path) {
    auto names = ListNames(dir, budget_, kNoItem);
    if (!names) {
      return Pass(names);
    }
    std::vector<std::string> dirs;
    for (const std::string& name : *names) {
      struct stat status{};
      if (::fstatat(dir, name.c_str(), &status, AT_SYMLINK_NOFOLLOW) != 0) {
        return Fail(Rule::kFileType, "unreadable directory entry");
      }
      if (S_ISLNK(status.st_mode)) {
        return Fail(Rule::kFileType, "a symbolic link in the artifact");
      }
      if (S_ISDIR(status.st_mode)) {
        dirs.push_back(name);
        continue;
      }
      if (!S_ISREG(status.st_mode) || status.st_nlink != 1) {
        return Fail(Rule::kFileType, "not a regular, singly linked file");
      }
      if (auto ok = Trusted(status, path / name); !ok) {
        return Pass(ok);
      }
      present_.push_back(std::string(prefix) + name);
    }
    return dirs;
  }

  Step Walk() {
    const std::filesystem::path root = root_path_;
    auto dirs = Scan(artifact_.root_.get(), "", root);
    if (!dirs) {
      return Pass(dirs);
    }
    for (const std::string& d : *dirs) {
      if (d != "data" && d != "meta") {
        return Fail(Rule::kFileSet, "an unexpected directory");
      }
    }
    for (const std::string& d : *dirs) {
      FileDescriptor fd(::openat(artifact_.root_.get(), d.c_str(),
                                 O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
      if (!fd.valid()) {
        return Fail(Rule::kFileType, "unreadable directory");
      }
      struct stat status{};
      if (::fstat(fd.get(), &status) != 0) {
        return Fail(Rule::kIo, "fstat failed");
      }
      if (auto ok = Trusted(status, root / d); !ok) {
        return ok;
      }
      auto nested = Scan(fd.get(), d + "/", root / d);
      if (!nested) {
        return Pass(nested);
      }
      if (!nested->empty()) {
        return Fail(Rule::kFileSet, "an unexpected directory");
      }
      (d == "data" ? data_ : meta_) = std::move(fd);
    }
    std::ranges::sort(present_);
    return {};
  }

  // --- manifest.json (layout.py _verify and _check_manifest)

  Step ReadManifest() {
    auto bytes = ReadDocument(artifact_.root_.get(), "manifest.json", std::nullopt,
                              kMaxManifestBytes, kNoItem);
    if (!bytes) {
      return Pass(bytes);
    }
    manifest_bytes_ = std::move(*bytes);
    auto parsed = json::Parse(manifest_bytes_);
    if (!parsed) {
      return Pass(parsed);
    }
    manifest_ = std::move(*parsed);
    const json::Value m = manifest_.root();
    const auto format = m.find("format");
    if (!m.is_object() || !format || !schema::IsString(*format, kManifestFormat)) {
      return Fail(Rule::kFormat, "not a llmpalooza artifact");
    }
    const auto version = m.find("format_version");
    if (!version || !schema::IsInt(*version, 0)) {
      return Fail(Rule::kUnsupportedVersion, "format_version is not 0; re-import required");
    }
    const auto layout = m.find("layout");
    if (!layout || !IsProfile(*layout)) {
      return Fail(Rule::kUnsupportedProfile, "layout is not spark-v0; re-import required");
    }
    if (!manifest_.canonical()) {
      return Fail(Rule::kCanonical, "manifest.json is not in canonical form");
    }
    if (artifact_.id_ != base::ToHex(Sha256Of(manifest_bytes_))) {
      return Fail(Rule::kIdentity, "the artifact name is not the manifest digest");
    }
    return {};
  }

  static bool IsProfile(json::Value v) {
    if (!v.is_object() || v.size() != 7) {
      return false;
    }
    const auto get = [&](std::string_view key) { return v.find(key); };
    const auto profile = get("profile");
    const auto version = get("profile_version");
    const auto file_alignment = get("file_alignment");
    const auto chunk = get("chunk_bytes");
    const auto member = get("member_alignment");
    const auto container = get("container");
    const auto order = get("byte_order");
    return profile && schema::IsString(*profile, "spark-v0") && version &&
           schema::IsInt(*version, 0) && file_alignment &&
           schema::IsInt(*file_alignment, static_cast<std::int64_t>(kFileAlignment)) && chunk &&
           schema::IsInt(*chunk, static_cast<std::int64_t>(kChunkBytes)) && member &&
           schema::IsInt(*member, static_cast<std::int64_t>(kMemberAlignment)) && container &&
           schema::IsString(*container, "safetensors") && order &&
           schema::IsString(*order, "little");
  }

  Step CheckManifest() {
    const json::Value m = manifest_.root();
    if (auto ok = schema::Object(m,
                                 {"format", "format_version", "experimental", "layout", "model",
                                  "source", "transformations", "converter", "files"},
                                 {}, "manifest keys");
        !ok) {
      return ok;
    }
    if (schema::Get(m, "experimental").kind() != json::Kind::kTrue) {
      return Fail(Rule::kSchema, "experimental must be true before D-018's gate");
    }
    if (auto ok = CheckModel(schema::Get(m, "model")); !ok) {
      return ok;
    }
    if (auto ok = CheckSources(schema::Get(m, "source")); !ok) {
      return ok;
    }
    if (auto ok = CheckTransformations(schema::Get(m, "transformations")); !ok) {
      return ok;
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
    artifact_.converter_ = {.name = std::string(*name), .version = std::string(*version)};
    return CheckFiles(schema::Get(m, "files"));
  }

  Step CheckModel(json::Value model) {
    if (auto ok = schema::Object(model, {"architecture", "expert_count", "representation"}, {},
                                 "model keys");
        !ok) {
      return ok;
    }
    auto arch =
        schema::Str(schema::Get(model, "architecture"), schema::IsName, "model.architecture");
    if (!arch) {
      return Pass(arch);
    }
    auto experts = schema::Int(schema::Get(model, "expert_count"), 0,
                               static_cast<std::int64_t>(kMaxExperts), "model.expert_count");
    if (!experts) {
      return Pass(experts);
    }
    const json::Value reps = schema::Get(model, "representation");
    if (auto ok = schema::List(reps, 1, "model.representation"); !ok) {
      return ok;
    }
    std::vector<std::string_view> names;
    for (std::size_t i = 0; i < reps.size(); ++i) {
      if (!reps.at(i).is_string()) {
        return Fail(Rule::kSchema, "model.representation");
      }
      names.push_back(reps.at(i).string());
    }
    for (std::size_t i = 0; i < names.size(); ++i) {
      if ((i > 0 && names[i - 1] >= names[i]) ||
          (names[i] != "ggml" && names[i] != "exl3" && names[i] != "plain")) {
        return Fail(Rule::kSchema, "model.representation");
      }
      artifact_.model_.representation.push_back(FamilyNamed(names[i]));
      representation_names_.push_back(names[i]);
    }
    artifact_.model_.architecture = std::string(*arch);
    artifact_.model_.expert_count = static_cast<std::uint32_t>(*experts);
    return {};
  }

  static Family FamilyNamed(std::string_view name) {
    if (name == "ggml") {
      return Family::kGgml;
    }
    return name == "exl3" ? Family::kExl3 : Family::kPlain;
  }

  Step CheckSources(json::Value sources) {
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
      artifact_.sources_.push_back({.name = std::string(*name),
                                    .bytes = Bytes(*bytes),
                                    .sha256 = schema::DigestFromHex(*sha).value_or(Digest{})});
    }
    for (std::size_t i = 1; i < artifact_.sources_.size(); ++i) {
      if (artifact_.sources_[i - 1].name >= artifact_.sources_[i].name) {
        return Fail(Rule::kCanonical, "source not in canonical order or duplicated", i);
      }
    }
    return {};
  }

  Step CheckTransformations(json::Value list) {
    if (auto ok = schema::List(list, 0, "transformations"); !ok) {
      return ok;
    }
    for (std::size_t i = 0; i < list.size(); ++i) {
      const json::Value t = list.at(i);
      const auto kind = t.is_object() ? t.find("kind") : std::nullopt;
      const std::string_view k = kind && kind->is_string() ? kind->string() : "";
      if (k == "dedupe-identical") {
        if (auto ok = schema::Object(t, {"kind", "resource", "role", "sha256"}, {},
                                     "transformation keys", i);
            !ok) {
          return ok;
        }
        auto resource =
            schema::Str(schema::Get(t, "resource"), schema::IsName, "transformation.resource", i);
        if (!resource) {
          return Pass(resource);
        }
        auto role = schema::Str(schema::Get(t, "role"), schema::IsName, "transformation.role", i);
        if (!role) {
          return Pass(role);
        }
        auto sha = schema::Str(schema::Get(t, "sha256"), schema::IsHex, "transformation.sha256", i);
        if (!sha) {
          return Pass(sha);
        }
        dedupes_.emplace_back(*resource, *role);
      } else if (k == "expert-slice") {
        if (auto ok = schema::Object(t, {"kind", "tensor", "count"}, {}, "transformation keys", i);
            !ok) {
          return ok;
        }
        auto tensor =
            schema::Str(schema::Get(t, "tensor"), schema::IsName, "transformation.tensor", i);
        if (!tensor) {
          return Pass(tensor);
        }
        auto count = schema::Int(schema::Get(t, "count"), 1, static_cast<std::int64_t>(kMaxExperts),
                                 "transformation.count", i);
        if (!count) {
          return Pass(count);
        }
        slices_.emplace_back(*tensor, *count);
      } else {
        return Fail(Rule::kSchema, "unknown transformation kind", i);
      }
    }
    // Canonical order: by each element's canonical JSON (layout.py _canon_key).
    std::string previous;
    std::string current;
    for (std::size_t i = 0; i < list.size(); ++i) {
      current.clear();
      json::Serialize(list.at(i), current);
      if (i > 0 && previous >= current) {
        return Fail(Rule::kCanonical, "transformations not in canonical order or duplicated", i);
      }
      std::swap(previous, current);
    }
    return {};
  }

  Step CheckFiles(json::Value list) {
    if (auto ok = schema::List(list, 2, "files"); !ok) {
      return ok;
    }
    for (std::size_t i = 0; i < list.size(); ++i) {
      const json::Value f = list.at(i);
      if (auto ok = schema::Object(f, {"path", "role", "bytes", "sha256"}, {}, "file keys", i);
          !ok) {
        return ok;
      }
      auto path = schema::Str(schema::Get(f, "path"), schema::AnyString, "file.path", i);
      if (!path) {
        return Pass(path);
      }
      auto role = schema::Str(schema::Get(f, "role"), schema::AnyString, "file.role", i);
      if (!role) {
        return Pass(role);
      }
      FileRole kind = FileRole::kIndex;
      bool fits = false;
      if (*role == "index") {
        fits = *path == "index.json";
      } else if (*role == "shard") {
        kind = FileRole::kShard;
        fits = schema::IsShardPath(*path);
      } else if (*role == "source-metadata") {
        kind = FileRole::kSourceMetadata;
        fits = schema::IsMetaPath(*path);
      }
      if (!fits) {
        return Fail(Rule::kPath, "a listed path does not fit its role", i);
      }
      auto bytes = schema::Int(schema::Get(f, "bytes"), 0, schema::kMaxInt, "file.bytes", i);
      if (!bytes) {
        return Pass(bytes);
      }
      auto sha = schema::Str(schema::Get(f, "sha256"), schema::IsHex, "file.sha256", i);
      if (!sha) {
        return Pass(sha);
      }
      artifact_.files_.push_back({.path = std::string(*path),
                                  .role = kind,
                                  .bytes = Bytes(*bytes),
                                  .sha256 = schema::DigestFromHex(*sha).value_or(Digest{})});
    }
    const auto& files = artifact_.files_;
    for (std::size_t i = 1; i < files.size(); ++i) {
      if (files[i - 1].path >= files[i].path) {
        return Fail(Rule::kCanonical, "files not in canonical order or duplicated", i);
      }
    }
    if (std::ranges::count(files, FileRole::kIndex, &ListedFile::role) != 1) {
      return Fail(Rule::kFileSet, "exactly one index.json is required");
    }
    return {};
  }

  const ListedFile* FindFile(std::string_view path) const {
    const auto& files = artifact_.files_;
    const auto found = std::ranges::lower_bound(files, path, {}, &ListedFile::path);
    return found != files.end() && found->path == path ? &*found : nullptr;
  }

  // --- the listed files against what is there

  Step CheckFileSet() {
    std::vector<std::string> listed;
    listed.reserve(artifact_.files_.size() + 1);
    for (const ListedFile& f : artifact_.files_) {
      listed.push_back(f.path);
    }
    listed.emplace_back("manifest.json");
    std::ranges::sort(listed);
    if (listed != present_) {
      return Fail(Rule::kFileSet, "files present and listed differ");
    }
    // Sizes first: nothing below trusts a declared length.
    for (std::size_t i = 0; i < artifact_.files_.size(); ++i) {
      const ListedFile& f = artifact_.files_[i];
      // Through the directories the walk opened, never re-resolving a path.
      int dir = artifact_.root_.get();
      std::string_view name = f.path;
      if (name.starts_with("data/") || name.starts_with("meta/")) {
        dir = name.starts_with("data/") ? data_.get() : meta_.get();
        name.remove_prefix(5);
      }
      struct stat status{};
      if (::fstatat(dir, std::string(name).c_str(), &status, AT_SYMLINK_NOFOLLOW) != 0) {
        return Fail(Rule::kIo, "fstatat failed", i);
      }
      if (static_cast<std::uint64_t>(status.st_size) != f.bytes.value()) {
        return Fail(Rule::kFileSize, "size differs from the manifest", i);
      }
    }
    return {};
  }

  // Kept metadata: within its caps, and each file a recorded source (or,
  // for <stem>.kv.gguf, kept from a recorded <stem>.gguf). Contents are not
  // read here (see artifact.h).
  Step CheckMetadata() {
    std::uint64_t total = 0;
    const auto source = [&](std::string_view name) -> const Source* {
      const auto& sources = artifact_.sources_;
      const auto found = std::ranges::lower_bound(sources, name, {}, &Source::name);
      return found != sources.end() && found->name == name ? &*found : nullptr;
    };
    for (std::size_t i = 0; i < artifact_.files_.size(); ++i) {
      const ListedFile& f = artifact_.files_[i];
      if (f.role != FileRole::kSourceMetadata) {
        continue;
      }
      total = SatAdd(total, f.bytes.value());
      if (f.bytes.value() > kMaxMetadataBytes || total > kMaxMetadataTotal) {
        return Fail(Rule::kFileSize, "kept metadata over its caps", i);
      }
      const std::string_view name = std::string_view(f.path).substr(5);
      constexpr std::string_view kKv = ".kv.gguf";
      if (name.ends_with(kKv)) {
        const std::string stem = std::string(name.substr(0, name.size() - kKv.size())) + ".gguf";
        if (source(stem) == nullptr) {
          return Fail(Rule::kFileSet, "kept GGUF metadata without its GGUF source", i);
        }
      } else {
        const Source* s = source(name);
        if (s == nullptr || s->bytes != f.bytes || s->sha256 != f.sha256) {
          return Fail(Rule::kFileSet, "kept metadata is not a verbatim recorded source", i);
        }
      }
    }
    return {};
  }

  // --- index.json (layout.py _check_index)

  Step ReadIndex() {
    const ListedFile* file = FindFile("index.json");
    auto bytes = ReadDocument(artifact_.root_.get(), "index.json", file->bytes.value(),
                              kMaxIndexBytes, kNoItem);
    if (!bytes) {
      return Pass(bytes);
    }
    index_bytes_ = std::move(*bytes);
    if (Sha256Of(index_bytes_) != file->sha256) {
      return Fail(Rule::kHash, "index.json differs from its manifest digest");
    }
    auto parsed = json::Parse(index_bytes_);
    if (!parsed) {
      return Pass(parsed);
    }
    index_ = std::move(*parsed);
    if (!index_.canonical()) {
      return Fail(Rule::kCanonical, "index.json is not in canonical form");
    }
    return {};
  }

  Step CheckIndex() {
    const json::Value index = index_.root();
    if (auto ok = schema::Object(
            index,
            {"format", "format_version", "file_alignment", "chunk_bytes", "member_alignment",
             "shards", "groups", "resources", "expert_arrays", "chunk_sha256"},
            {}, "index keys");
        !ok) {
      return ok;
    }
    if (!schema::IsString(schema::Get(index, "format"), kIndexFormat) ||
        !schema::IsInt(schema::Get(index, "format_version"), 0) ||
        !schema::IsInt(schema::Get(index, "file_alignment"),
                       static_cast<std::int64_t>(kFileAlignment)) ||
        !schema::IsInt(schema::Get(index, "chunk_bytes"), static_cast<std::int64_t>(kChunkBytes)) ||
        !schema::IsInt(schema::Get(index, "member_alignment"),
                       static_cast<std::int64_t>(kMemberAlignment))) {
      return Fail(Rule::kUnsupportedVersion, "index header differs from the profile");
    }
    for (auto step : {&Opener::CheckShardList, &Opener::CheckGroups, &Opener::CheckChunkHashes,
                      &Opener::CheckResources, &Opener::CheckExpertArrays, &Opener::CheckBindings,
                      &Opener::CheckPlacement, &Opener::CheckTransformationsMatch}) {
      if (auto ok = (this->*step)(); !ok) {
        return ok;
      }
    }
    return {};
  }

  Step CheckShardList() {
    const json::Value shards = schema::Get(index_.root(), "shards");
    if (auto ok = schema::List(shards, 1, "shards"); !ok) {
      return ok;
    }
    auto& out = artifact_.layout_.shards;
    for (std::size_t i = 0; i < shards.size(); ++i) {
      const json::Value s = shards.at(i);
      if (auto ok = schema::Object(s, {"path", "data_offset", "data_bytes", "header_sha256"}, {},
                                   "shard keys", i);
          !ok) {
        return ok;
      }
      auto path = schema::Str(schema::Get(s, "path"), schema::IsShardPath, "shard.path", i);
      if (!path) {
        return Pass(path);
      }
      if (*path != ShardPath(i)) {
        return Fail(Rule::kFileSet, "shard path out of sequence", i);
      }
      auto offset =
          schema::Int(schema::Get(s, "data_offset"), static_cast<std::int64_t>(kFileAlignment),
                      schema::kMaxInt, "shard.data_offset", i);
      if (!offset) {
        return Pass(offset);
      }
      if (*offset % kFileAlignment != 0) {
        return Fail(Rule::kBounds, "shard alignment", i);
      }
      auto bytes =
          schema::Int(schema::Get(s, "data_bytes"), static_cast<std::int64_t>(kFileAlignment),
                      schema::kMaxInt, "shard.data_bytes", i);
      if (!bytes) {
        return Pass(bytes);
      }
      if (*bytes % kFileAlignment != 0) {
        return Fail(Rule::kBounds, "shard alignment", i);
      }
      auto sha =
          schema::Str(schema::Get(s, "header_sha256"), schema::IsHex, "shard.header_sha256", i);
      if (!sha) {
        return Pass(sha);
      }
      const ListedFile* file = FindFile(*path);
      if (file == nullptr || file->role != FileRole::kShard ||
          file->bytes.value() != *offset + *bytes) {
        return Fail(Rule::kFileSet, "shard not listed or of the wrong size", i);
      }
      out.push_back({.path = std::string(*path),
                     .data_offset = Bytes(*offset),
                     .data_bytes = Bytes(*bytes),
                     .header_sha256 = schema::DigestFromHex(*sha).value_or(Digest{})});
    }
    const auto listed = std::ranges::count(artifact_.files_, FileRole::kShard, &ListedFile::role);
    if (static_cast<std::size_t>(listed) != out.size()) {
      return Fail(Rule::kFileSet, "listed shards and indexed shards differ");
    }
    return {};
  }

  static std::string ShardPath(std::size_t i) {
    std::string digits = std::to_string(i);
    if (digits.size() < 5) {
      digits.insert(0, 5 - digits.size(), '0');
    }
    return "data/" + digits + ".safetensors";
  }

  Step CheckGroups() {
    const json::Value groups = schema::Get(index_.root(), "groups");
    if (auto ok = schema::List(groups, 1, "groups"); !ok) {
      return ok;
    }
    const auto& shards = artifact_.layout_.shards;
    std::vector<std::uint64_t> cursor(shards.size(), 0);
    std::uint64_t next_chunk = 0;
    std::uint64_t last_shard = 0;
    auto& out = artifact_.layout_.groups;
    for (std::size_t gid = 0; gid < groups.size(); ++gid) {
      const json::Value g = groups.at(gid);
      if (auto ok = schema::Object(g,
                                   {"kind", "layer", "expert", "shard", "offset", "stored_bytes",
                                    "used_bytes", "first_chunk"},
                                   {}, "group keys", gid);
          !ok) {
        return ok;
      }
      const json::Value kind_value = schema::Get(g, "kind");
      const std::string_view kind_name = kind_value.is_string() ? kind_value.string() : "";
      Group group;
      if (kind_name == "table") {
        group.kind = GroupKind::kTable;
      } else if (kind_name == "layer") {
        group.kind = GroupKind::kLayer;
      } else if (kind_name == "expert") {
        group.kind = GroupKind::kExpert;
      } else if (kind_name == "global") {
        group.kind = GroupKind::kGlobal;
      } else if (kind_name == "head") {
        group.kind = GroupKind::kHead;
      } else {
        return Fail(Rule::kSchema, "group kind", gid);
      }
      const json::Value layer = schema::Get(g, "layer");
      if (group.kind == GroupKind::kLayer || group.kind == GroupKind::kExpert) {
        auto n = schema::Int(layer, 0, static_cast<std::int64_t>(kMaxLayer), "group layer", gid);
        if (!n) {
          return Pass(n);
        }
        group.layer = static_cast<std::uint32_t>(*n);
      } else if (layer.kind() != json::Kind::kNull) {
        return Fail(Rule::kSchema, "group layer", gid);
      }
      const json::Value expert = schema::Get(g, "expert");
      if (group.kind == GroupKind::kExpert) {
        auto n =
            schema::Int(expert, 0, static_cast<std::int64_t>(kMaxExperts), "group expert", gid);
        if (!n) {
          return Pass(n);
        }
        group.expert = static_cast<std::uint32_t>(*n);
        if (!expert_seen_.emplace(group.layer.value_or(0), *n).second) {
          return Fail(Rule::kExpertArray, "duplicate expert group", gid);
        }
      } else if (expert.kind() != json::Kind::kNull) {
        return Fail(Rule::kSchema, "group expert", gid);
      }
      auto shard = schema::Int(schema::Get(g, "shard"), 0,
                               static_cast<std::int64_t>(shards.size()) - 1, "group shard", gid);
      if (!shard) {
        return Pass(shard);
      }
      if (*shard < last_shard) {
        return Fail(Rule::kBounds, "group shard order", gid);
      }
      last_shard = *shard;
      auto stored =
          schema::Int(schema::Get(g, "stored_bytes"), static_cast<std::int64_t>(kFileAlignment),
                      schema::kMaxInt, "group stored_bytes", gid);
      if (!stored) {
        return Pass(stored);
      }
      auto used =
          schema::Int(schema::Get(g, "used_bytes"), 1, schema::kMaxInt, "group used_bytes", gid);
      if (!used) {
        return Pass(used);
      }
      auto offset = schema::Int(schema::Get(g, "offset"), 0, schema::kMaxInt, "group offset", gid);
      if (!offset) {
        return Pass(offset);
      }
      if (*offset != cursor[*shard] || *stored != AlignUp(*used, kFileAlignment)) {
        return Fail(Rule::kBounds, "group offset or size", gid);
      }
      auto first =
          schema::Int(schema::Get(g, "first_chunk"), 0, schema::kMaxInt, "group first_chunk", gid);
      if (!first) {
        return Pass(first);
      }
      if (*first != next_chunk) {
        return Fail(Rule::kBounds, "group chunk numbering", gid);
      }
      cursor[*shard] = SatAdd(cursor[*shard], *stored);
      next_chunk = SatAdd(next_chunk, Chunks(*stored));
      group.shard = static_cast<std::uint32_t>(*shard);
      group.offset = Bytes(*offset);
      group.used = Bytes(*used);
      group.stored = Bytes(*stored);
      group.first_chunk = *first;
      out.push_back(group);
    }
    for (std::size_t i = 0; i < shards.size(); ++i) {
      if (cursor[i] != shards[i].data_bytes.value()) {
        return Fail(Rule::kBounds, "groups do not tile the shard", i);
      }
    }
    total_chunks_ = next_chunk;
    return {};
  }

  Step CheckChunkHashes() {
    const json::Value hashes = schema::Get(index_.root(), "chunk_sha256");
    if (auto ok = schema::List(hashes, 0, "chunk_sha256"); !ok) {
      return ok;
    }
    if (hashes.size() != total_chunks_) {
      return Fail(Rule::kSchema, "chunk_sha256 count differs from the chunks");
    }
    artifact_.chunk_sha256_.reserve(hashes.size());
    for (std::size_t i = 0; i < hashes.size(); ++i) {
      const json::Value h = hashes.at(i);
      if (!h.is_string() || !schema::IsHex(h.string())) {
        return Fail(Rule::kSchema, "chunk_sha256", i);
      }
      artifact_.chunk_sha256_.push_back(schema::DigestFromHex(h.string()).value_or(Digest{}));
    }
    // Per-group chunk counts are now known to fit (the list bounds them).
    for (Group& g : artifact_.layout_.groups) {
      g.chunks = static_cast<std::uint32_t>(Chunks(g.stored.value()));
    }
    return {};
  }

  Step Claim(std::string_view name, Item item) {
    if (!names_.insert(name).second) {
      return Fail(Rule::kSchema, "duplicate name", item);
    }
    return {};
  }

  Step CheckResources() {
    const json::Value resources = schema::Get(index_.root(), "resources");
    if (auto ok = schema::List(resources, 0, "resources"); !ok) {
      return ok;
    }
    const json::Value arrays = schema::Get(index_.root(), "expert_arrays");
    if (auto ok = schema::List(arrays, 0, "expert_arrays"); !ok) {
      return ok;
    }
    const std::size_t groups = artifact_.layout_.groups.size();
    if (groups > kMaxEntries || resources.size() + arrays.size() > kMaxEntries) {
      return Fail(Rule::kBounds, "more entries than the cap");
    }
    entries_ = resources.size();
    for (std::size_t i = 0; i < resources.size(); ++i) {
      if (auto ok = CheckResource(resources.at(i), i); !ok) {
        return ok;
      }
    }
    const auto& out = artifact_.resources_;
    for (std::size_t i = 1; i < out.size(); ++i) {
      if (std::pair(out[i - 1].group, out[i - 1].offset) >=
          std::pair(out[i].group, out[i].offset)) {
        return Fail(Rule::kCanonical, "resources not in canonical order or duplicated", i);
      }
    }
    return {};
  }

  Step CheckResource(json::Value r, std::size_t i) {
    if (auto ok = schema::Object(
            r, {"name", "group", "offset", "bytes", "readable_bytes", "roles", "repr"}, {"access"},
            "resource keys", i);
        !ok) {
      return ok;
    }
    auto name = schema::Str(schema::Get(r, "name"), schema::IsName, "resource.name", i);
    if (!name) {
      return Pass(name);
    }
    if (*name == "__metadata__") {
      return Fail(Rule::kSchema, "reserved name", i);
    }
    if (auto ok = Claim(*name, i); !ok) {
      return ok;
    }
    const auto& groups = artifact_.layout_.groups;
    auto gid = schema::Int(schema::Get(r, "group"), 0, static_cast<std::int64_t>(groups.size()) - 1,
                           "resource.group", i);
    if (!gid) {
      return Pass(gid);
    }
    if (groups[*gid].kind == GroupKind::kExpert) {
      return Fail(Rule::kExpertArray, "a resource placed in an expert group", i);
    }
    const json::Value roles = schema::Get(r, "roles");
    if (auto ok = schema::List(roles, 1, "resource.roles", i); !ok) {
      return ok;
    }
    if (!schema::IsString(roles.at(0), *name)) {
      return Fail(Rule::kSchema, "roles[0] must be the name", i);
    }
    if (roles.size() > 1 + kMaxAliases) {
      return Fail(Rule::kBounds, "more alias roles than the cap", i);
    }
    entries_ += roles.size() - 1;
    if (entries_ > kMaxEntries) {
      return Fail(Rule::kBounds, "more entries than the cap", i);
    }
    for (std::size_t k = 0; k < roles.size(); ++k) {
      if (!roles.at(k).is_string()) {
        return Fail(Rule::kCanonical, "alias roles must be sorted and unique", i);
      }
    }
    for (std::size_t k = 2; k < roles.size(); ++k) {
      if (roles.at(k - 1).string() >= roles.at(k).string()) {
        return Fail(Rule::kCanonical, "alias roles must be sorted and unique", i);
      }
    }
    Resource out;
    out.name = std::string(*name);
    out.group = static_cast<std::uint32_t>(*gid);
    for (std::size_t k = 0; k < roles.size(); ++k) {
      const std::string_view role = roles.at(k).string();
      if (!schema::IsName(role) || role == "__metadata__") {
        return Fail(Rule::kSchema, "bad role", i);
      }
      if (!roles_.insert(role).second) {
        return Fail(Rule::kSchema, "a role bound twice", i);
      }
      out.roles.emplace_back(role);
    }
    const auto access = r.find("access");
    if (access && !schema::IsString(*access, "rows")) {
      return Fail(Rule::kSchema, "access must be rows", i);
    }
    auto rep = ParseRepresentation(schema::Get(r, "repr"));
    if (!rep) {
      return Pass(Error{.rule = rep.error().rule, .reason = rep.error().reason, .item = i});
    }
    families_.insert(FamilyName(rep->family));
    auto bytes = schema::Int(schema::Get(r, "bytes"), 1, schema::kMaxInt, "resource.bytes", i);
    if (!bytes) {
      return Pass(bytes);
    }
    if (*bytes != rep->bytes) {
      return Fail(Rule::kRepr, "bytes differ from what the representation implies", i);
    }
    auto readable = schema::Int(schema::Get(r, "readable_bytes"), 1, schema::kMaxInt,
                                "resource.readable_bytes", i);
    if (!readable) {
      return Pass(readable);
    }
    if (*readable != rep->readable) {
      return Fail(Rule::kRepr, "readable_bytes differ from the backend's over-read rule", i);
    }
    if (access && !RowGeometryOf(*rep)) {
      return Fail(Rule::kRepr, "a row table must be 2-D", i);
    }
    auto offset = schema::Int(schema::Get(r, "offset"), 0, schema::kMaxInt, "resource.offset", i);
    if (!offset) {
      return Pass(offset);
    }
    out.offset = Bytes(*offset);
    out.bytes = Bytes(*bytes);
    out.readable = Bytes(*readable);
    out.rows = access.has_value();
    spans_.push_back({.group = out.group,
                      .offset = *offset,
                      .bytes = *bytes,
                      .readable = *readable,
                      .name = out.name,
                      .view = ContainerViewOf(*rep)});
    out.repr = std::move(*rep);
    artifact_.resources_.push_back(std::move(out));
    return {};
  }

  static std::unexpected<Error> Pass(const Error& error) { return std::unexpected(error); }
  template <typename T>
  static std::unexpected<Error> Pass(const std::expected<T, Error>& failed) {
    return std::unexpected(failed.error());
  }

  Step CheckExpertArrays() {
    const json::Value arrays = schema::Get(index_.root(), "expert_arrays");
    const auto& groups = artifact_.layout_.groups;
    for (std::size_t i = 0; i < arrays.size(); ++i) {
      const json::Value a = arrays.at(i);
      if (auto ok = schema::Object(a,
                                   {"name", "layer", "count", "first_group", "group_offset",
                                    "slice_bytes", "readable_bytes", "repr"},
                                   {}, "expert array keys", i);
          !ok) {
        return ok;
      }
      auto name = schema::Str(schema::Get(a, "name"), schema::IsName, "expert_array.name", i);
      if (!name) {
        return Pass(name);
      }
      if (*name == "__metadata__") {
        return Fail(Rule::kSchema, "reserved name", i);
      }
      if (auto ok = Claim(*name, i); !ok) {
        return ok;
      }
      auto layer = schema::Int(schema::Get(a, "layer"), 0, static_cast<std::int64_t>(kMaxLayer),
                               "expert_array.layer", i);
      if (!layer) {
        return Pass(layer);
      }
      auto count = schema::Int(schema::Get(a, "count"), 1, static_cast<std::int64_t>(kMaxExperts),
                               "expert_array.count", i);
      if (!count) {
        return Pass(count);
      }
      entries_ += *count;
      if (entries_ > kMaxEntries) {
        return Fail(Rule::kBounds, "more resources and expert slices than the cap", i);
      }
      if (*count != artifact_.model_.expert_count) {
        return Fail(Rule::kExpertArray, "count differs from model.expert_count", i);
      }
      auto first =
          schema::Int(schema::Get(a, "first_group"), 0,
                      static_cast<std::int64_t>(groups.size()) - static_cast<std::int64_t>(*count),
                      "expert_array.first_group", i);
      if (!first) {
        return Pass(first);
      }
      auto rep = ParseRepresentation(schema::Get(a, "repr"));
      if (!rep) {
        return Pass(Error{.rule = rep.error().rule, .reason = rep.error().reason, .item = i});
      }
      if (rep->family != Family::kGgml || rep->dims.size() != 2) {
        return Fail(Rule::kExpertArray, "a slice must be a 2-D ggml matrix", i);
      }
      families_.insert("ggml");
      auto slice = schema::Int(schema::Get(a, "slice_bytes"), 1, schema::kMaxInt,
                               "expert_array.slice_bytes", i);
      if (!slice) {
        return Pass(slice);
      }
      if (*slice != rep->bytes) {
        return Fail(Rule::kRepr, "slice sizes disagree with the representation", i);
      }
      auto readable = schema::Int(schema::Get(a, "readable_bytes"), 1, schema::kMaxInt,
                                  "expert_array.readable_bytes", i);
      if (!readable) {
        return Pass(readable);
      }
      if (*readable != rep->readable) {
        return Fail(Rule::kRepr, "slice sizes disagree with the representation", i);
      }
      const Group& g0 = groups[*first];
      for (std::uint64_t e = 0; e < *count; ++e) {
        const Group& g = groups[*first + e];
        if (g.kind != GroupKind::kExpert || g.layer != *layer || g.expert != e ||
            g.stored != g0.stored) {
          return Fail(Rule::kExpertArray, "groups are not uniform expert groups", i);
        }
      }
      auto& layer_arrays = arrays_by_layer_[*layer];
      layer_arrays.emplace_back(*first, *count);
      auto group_offset = schema::Int(schema::Get(a, "group_offset"), 0, schema::kMaxInt,
                                      "expert_array.group_offset", i);
      if (!group_offset) {
        return Pass(group_offset);
      }
      const ContainerView view = ContainerViewOf(*rep);
      for (std::uint64_t e = 0; e < *count; ++e) {
        spans_.push_back({.group = static_cast<std::uint32_t>(*first + e),
                          .offset = *group_offset,
                          .bytes = *slice,
                          .readable = *readable,
                          .name = std::string(*name) + "#" + std::to_string(e),
                          .view = view});
      }
      artifact_.expert_arrays_.push_back({.name = std::string(*name),
                                          .layer = static_cast<std::uint32_t>(*layer),
                                          .count = static_cast<std::uint32_t>(*count),
                                          .first_group = static_cast<std::uint32_t>(*first),
                                          .group_offset = Bytes(*group_offset),
                                          .slice_bytes = Bytes(*slice),
                                          .readable = Bytes(*readable),
                                          .repr = std::move(*rep)});
    }
    const auto& out = artifact_.expert_arrays_;
    for (std::size_t i = 1; i < out.size(); ++i) {
      if (std::pair(out[i - 1].first_group, out[i - 1].group_offset) >=
          std::pair(out[i].first_group, out[i].group_offset)) {
        return Fail(Rule::kCanonical, "expert_arrays not in canonical order or duplicated", i);
      }
    }
    return {};
  }

  // Roles against array names, expert coverage, EXL3 closure and the
  // manifest's families.
  Step CheckBindings() {
    for (const ExpertArray& a : artifact_.expert_arrays_) {
      if (roles_.contains(a.name)) {
        return Fail(Rule::kSchema, "a role names an expert array");
      }
    }
    std::vector<std::pair<std::uint64_t, std::uint64_t>> covered_want;
    for (const auto& [layer, arrays] : arrays_by_layer_) {
      for (const auto& a : arrays) {
        if (a != arrays.front()) {
          return Fail(Rule::kExpertArray, "a layer's arrays disagree on their expert groups",
                      layer);
        }
      }
      for (std::uint64_t e = 0; e < arrays.front().second; ++e) {
        covered_want.emplace_back(layer, e);
      }
    }
    const std::vector<std::pair<std::uint64_t, std::uint64_t>> covered(expert_seen_.begin(),
                                                                       expert_seen_.end());
    std::ranges::sort(covered_want);
    if (covered != covered_want) {
      return Fail(Rule::kExpertArray, "expert groups without a covering array");
    }
    std::vector<std::pair<std::string_view, const Representation*>> bound;
    for (const Resource& r : artifact_.resources_) {
      for (const std::string& role : r.roles) {
        bound.emplace_back(role, &r.repr);
      }
    }
    for (const ExpertArray& a : artifact_.expert_arrays_) {
      bound.emplace_back(a.name, &a.repr);
    }
    if (!Exl3ClosureHolds(bound)) {
      return Fail(Rule::kRepr, "an incomplete or misnamed EXL3 closure");
    }
    const std::vector<std::string_view> families(families_.begin(), families_.end());
    if (families != representation_names_) {
      return Fail(Rule::kSchema, "model.representation disagrees with the index");
    }
    return {};
  }

  // Aligned, non-overlapping readable ranges inside each group, the last
  // ending at used_bytes.
  Step CheckPlacement() {
    std::ranges::sort(spans_, SpanLess);
    const auto& groups = artifact_.layout_.groups;
    std::size_t s = 0;
    for (std::uint32_t gid = 0; gid < groups.size(); ++gid) {
      const Group& g = groups[gid];
      if (s >= spans_.size() || spans_[s].group != gid) {
        return Fail(Rule::kBounds, "an empty group", gid);
      }
      std::uint64_t end = 0;
      for (; s < spans_.size() && spans_[s].group == gid; ++s) {
        const Span& span = spans_[s];
        if (span.offset % kMemberAlignment != 0) {
          return Fail(Rule::kAlignment, "a resource off its member alignment", gid);
        }
        if (span.offset < end) {
          return Fail(Rule::kOverlap, "readable ranges overlap", gid);
        }
        // Both are at most 2^63 - 1, so the sum fits.
        if (span.offset + span.readable > g.stored.value()) {
          return Fail(Rule::kBounds, "a readable range outside its group", gid);
        }
        end = span.offset + span.readable;
      }
      if (end != g.used.value()) {
        return Fail(Rule::kBounds, "used_bytes is not the last readable end", gid);
      }
    }
    return {};
  }

  // The manifest's transformations describe the index exactly.
  Step CheckTransformationsMatch() {
    std::vector<std::pair<std::string_view, std::uint64_t>> want_slices;
    want_slices.reserve(artifact_.expert_arrays_.size());
    for (const ExpertArray& a : artifact_.expert_arrays_) {
      want_slices.emplace_back(a.name, a.count);
    }
    std::ranges::sort(want_slices);
    std::ranges::sort(slices_);
    if (slices_ != want_slices) {
      return Fail(Rule::kSchema, "expert-slice transformations disagree with the index");
    }
    std::vector<std::pair<std::string_view, std::string_view>> want_dedupes;
    for (const Resource& r : artifact_.resources_) {
      for (std::size_t k = 1; k < r.roles.size(); ++k) {
        want_dedupes.emplace_back(r.name, r.roles[k]);
      }
    }
    std::ranges::sort(want_dedupes);
    std::ranges::sort(dedupes_);
    if (dedupes_ != want_dedupes) {
      return Fail(Rule::kSchema, "dedupe transformations disagree with roles");
    }
    return {};
  }

  // --- shard headers (layout.py expected_shard_header and _check_shard)

  // The one header the index implies for shard `shard`: resources in
  // offset order, a single zero pad per gap numbered in order, then spaces
  // up to the 4 KiB-aligned data offset.
  std::expected<std::string, Error> ExpectedHeader(std::uint32_t shard) const {
    // The 8-byte length comes first; it is filled in at the end.
    std::string text(8, '\0');
    text += kShardPrefix;
    std::uint64_t size = text.size() - 8 + 1;  // the JSON, with its closing brace
    std::uint64_t pads = 0;
    std::string piece;
    const auto add = [&](std::string_view name, std::string_view dtype,
                         std::span<const std::uint64_t> shape, std::uint64_t start,
                         std::uint64_t end) -> Step {
      if (!schema::IsName(name) && !schema::IsSliceName(name) && !schema::IsPadName(name)) {
        return Fail(Rule::kSchema, "an unrepresentable entry name", shard);
      }
      piece.clear();
      piece += ",\"";
      piece += name;
      piece += R"(":{"dtype":")";
      piece += dtype;
      piece += R"(","shape":[)";
      for (std::size_t k = 0; k < shape.size(); ++k) {
        if (k > 0) {
          piece += ',';
        }
        piece += std::to_string(shape[k]);
      }
      piece += R"(],"data_offsets":[)";
      piece += std::to_string(start);
      piece += ',';
      piece += std::to_string(end);
      piece += "]}";
      size += piece.size();
      if (size > kMaxShardHeader) {
        return Fail(Rule::kContainer, "shard header over the upstream cap", shard);
      }
      text += piece;
      return {};
    };
    const auto pad = [&](std::uint64_t start, std::uint64_t end) -> Step {
      const std::string name = "~pad." + std::to_string(pads++);
      const std::array<std::uint64_t, 1> shape = {end - start};
      return add(name, "U8", shape, start, end);
    };
    const auto& groups = artifact_.layout_.groups;
    const auto first_span = [&](std::uint32_t gid) {
      return std::ranges::lower_bound(spans_, gid, {}, &Span::group);
    };
    // Groups are in shard order (CheckGroups), so a shard's are contiguous.
    const auto first_group = std::ranges::lower_bound(groups, shard, {}, &Group::shard);
    for (auto gid = static_cast<std::uint32_t>(first_group - groups.begin());
         gid < groups.size() && groups[gid].shard == shard; ++gid) {
      const Group& g = groups[gid];
      std::uint64_t cursor = g.offset.value();
      for (auto it = first_span(gid); it != spans_.end() && it->group == gid; ++it) {
        const std::uint64_t at = g.offset.value() + it->offset;
        if (at > cursor) {
          if (auto ok = pad(cursor, at); !ok) {
            return Pass(ok);
          }
        }
        if (auto ok = add(it->name, it->view.dtype, it->view.shape, at, at + it->bytes); !ok) {
          return Pass(ok);
        }
        cursor = at + it->bytes;
      }
      const std::uint64_t end = g.offset.value() + g.stored.value();
      if (end > cursor) {
        if (auto ok = pad(cursor, end); !ok) {
          return Pass(ok);
        }
      }
    }
    text += '}';
    const std::uint64_t data_offset = AlignUp(text.size(), kFileAlignment);
    if (data_offset - 8 > kMaxShardHeader) {
      return Fail(Rule::kContainer, "shard header over the upstream cap", shard);
    }
    const std::uint64_t length = data_offset - 8;
    for (std::size_t k = 0; k < 8; ++k) {
      text[k] = static_cast<char>((length >> (8U * k)) & 0xFFU);
    }
    text.resize(data_offset, ' ');
    return text;
  }

  Step CheckShards() {
    const auto& shards = artifact_.layout_.shards;
    for (std::uint32_t i = 0; i < shards.size(); ++i) {
      const Shard& s = shards[i];
      auto expected = ExpectedHeader(i);
      if (!expected) {
        return Pass(expected);
      }
      if (expected->size() != s.data_offset.value()) {
        return Fail(Rule::kContainer, "the data offset is not the canonical one", i);
      }
      const ListedFile* file = FindFile(s.path);
      struct stat status{};
      auto fd = OpenRegular(data_.get(), s.path.c_str() + 5, status, i);
      if (!fd) {
        return Pass(fd);
      }
      if (static_cast<std::uint64_t>(status.st_size) != file->bytes.value()) {
        return Fail(Rule::kFileSize, "size differs from the manifest", i);
      }
      base::Sha256 hash;
      std::string got(std::min<std::size_t>(expected->size(), kReadPiece), '\0');
      for (std::size_t at = 0; at < expected->size(); at += got.size()) {
        const std::size_t n = std::min(got.size(), expected->size() - at);
        auto read = ReadAt(fd->get(), at, std::span(got).first(n), i);
        if (!read) {
          return Pass(read);
        }
        const std::string_view piece = std::string_view(got).substr(0, *read);
        if (*read != n || piece != std::string_view(*expected).substr(at, n)) {
          return Fail(Rule::kContainer, "the shard header disagrees with the index", i);
        }
        hash.Update(piece);
      }
      if (hash.Finish() != s.header_sha256) {
        return Fail(Rule::kHash, "the shard header region differs from its digest", i);
      }
      artifact_.shard_files_.push_back({.device = status.st_dev,
                                        .inode = status.st_ino,
                                        .generation = Generation(fd->get()),
                                        .size = static_cast<std::uint64_t>(status.st_size),
                                        .changed_seconds = status.st_ctim.tv_sec,
                                        .changed_nanoseconds = status.st_ctim.tv_nsec});
    }
    return {};
  }

  void Finish() {
    auto& roles = artifact_.roles_;
    for (std::uint32_t i = 0; i < artifact_.resources_.size(); ++i) {
      for (const std::string& role : artifact_.resources_[i].roles) {
        roles.emplace_back(role, i);
      }
    }
    std::ranges::sort(roles);
    auto& arrays = artifact_.array_names_;
    for (std::uint32_t i = 0; i < artifact_.expert_arrays_.size(); ++i) {
      arrays.emplace_back(artifact_.expert_arrays_[i].name, i);
    }
    std::ranges::sort(arrays);
  }

  std::filesystem::path root_path_;
  OpenOptions options_;
  Artifact artifact_;
  struct stat trusted_root_{};
  FileDescriptor data_;
  FileDescriptor meta_;
  std::size_t budget_ = kMaxDirectoryEntries;
  std::vector<std::string> present_;
  std::string manifest_bytes_;
  json::Document manifest_;
  std::string index_bytes_;
  json::Document index_;
  std::vector<std::string_view> representation_names_;
  std::vector<std::pair<std::string_view, std::string_view>> dedupes_;
  std::vector<std::pair<std::string_view, std::uint64_t>> slices_;
  std::uint64_t total_chunks_ = 0;
  std::uint64_t entries_ = 0;
  // Ordered sets, not hash sets: their keys come from the artifact, which
  // could choose them to collide in a fixed hash and make each insert linear.
  std::set<std::string_view> names_;
  std::set<std::string_view> roles_;
  std::set<std::string_view> families_;
  std::set<std::pair<std::uint64_t, std::uint64_t>> expert_seen_;
  std::map<std::uint64_t, std::vector<std::pair<std::uint64_t, std::uint64_t>>> arrays_by_layer_;
  std::vector<Span> spans_;
};

std::expected<Artifact, Error> Artifact::Open(const std::filesystem::path& root,
                                              const OpenOptions& options) {
  Opener opener(root, options);
  return opener.Run();
}

std::optional<std::uint32_t> Artifact::FindResource(std::string_view role) const {
  const auto found = std::ranges::lower_bound(
      roles_, role, {}, [](const auto& entry) { return std::string_view(entry.first); });
  if (found == roles_.end() || found->first != role) {
    return std::nullopt;
  }
  return found->second;
}

std::optional<std::uint32_t> Artifact::FindExpertArray(std::string_view name) const {
  const auto found = std::ranges::lower_bound(
      array_names_, name, {}, [](const auto& entry) { return std::string_view(entry.first); });
  if (found == array_names_.end() || found->first != name) {
    return std::nullopt;
  }
  return found->second;
}

std::expected<Placement, Error> Artifact::ResourcePlacement(std::uint32_t resource) const {
  if (resource >= resources_.size()) {
    return Fail(Rule::kBounds, "no such resource", resource);
  }
  const Resource& r = resources_[resource];
  auto closure = ClosureOf(layout_, r.group, r.offset, r.readable);
  if (!closure) {
    return std::unexpected(closure.error());
  }
  return Placement{.group = r.group,
                   .offset = r.offset,
                   .bytes = r.bytes,
                   .readable = r.readable,
                   .closure = *closure};
}

std::expected<Placement, Error> Artifact::SlicePlacement(std::uint32_t array,
                                                         std::uint32_t expert) const {
  if (array >= expert_arrays_.size()) {
    return Fail(Rule::kBounds, "no such expert array", array);
  }
  const ExpertArray& a = expert_arrays_[array];
  if (expert >= a.count) {
    return Fail(Rule::kBounds, "no such expert", expert);
  }
  const std::uint32_t group = a.first_group + expert;
  auto closure = ClosureOf(layout_, group, a.group_offset, a.readable);
  if (!closure) {
    return std::unexpected(closure.error());
  }
  return Placement{.group = group,
                   .offset = a.group_offset,
                   .bytes = a.slice_bytes,
                   .readable = a.readable,
                   .closure = *closure};
}

std::expected<RowGeometry, Error> Artifact::Rows(std::uint32_t resource) const {
  if (resource >= resources_.size() || !resources_[resource].rows) {
    return Fail(Rule::kBounds, "not a row table", resource);
  }
  const auto geometry = RowGeometryOf(resources_[resource].repr);
  if (!geometry) {
    return Fail(Rule::kBounds, "not a row table", resource);  // unreachable once validated
  }
  return *geometry;
}

std::expected<std::vector<ChunkKey>, Error> Artifact::RowChunks(
    std::uint32_t resource, std::span<const std::uint64_t> rows) const {
  auto geometry = Rows(resource);
  if (!geometry) {
    return std::unexpected(geometry.error());
  }
  const Resource& r = resources_[resource];
  std::vector<ChunkKey> out;
  for (const std::uint64_t row : rows) {
    if (row >= geometry->rows) {
      return Fail(Rule::kBounds, "a row outside the table", row);
    }
    // row < rows, so row * row_bytes <= bytes.
    const Bytes start(r.offset.value() + (row * geometry->row_bytes));
    auto closure = ClosureOf(layout_, r.group, start, Bytes(geometry->row_bytes));
    if (!closure) {
      return std::unexpected(closure.error());
    }
    for (std::uint32_t k = closure->first; k <= closure->last; ++k) {
      out.push_back({.group = r.group, .chunk = k});
    }
  }
  std::ranges::sort(out);
  const auto [first, last] = std::ranges::unique(out);
  out.erase(first, last);
  return out;
}

std::expected<FileDescriptor, Error> Artifact::OpenShardForDirectRead(std::uint32_t shard) const {
  if (shard >= layout_.shards.size()) {
    return Fail(Rule::kBounds, "no such shard", shard);
  }
  FileDescriptor data(
      ::openat(root_.get(), "data", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
  if (!data.valid()) {
    return Fail(Rule::kFileType, "the data directory is gone or replaced", shard);
  }
  // Non-blocking until it is known to be the regular file validated at
  // open; then blocking, since io_uring would honour O_NONBLOCK.
  const auto opened =
      platform::OpenForDirectRead(data.get(), layout_.shards[shard].path.c_str() + 5);
  if (!opened) {
    return Fail(opened.error() == EINVAL ? Rule::kIo : Rule::kFileType,
                "cannot open the shard for direct reads", shard);
  }
  FileDescriptor fd(opened->fd);
  struct stat status{};
  if (::fstat(fd.get(), &status) != 0) {
    return Fail(Rule::kIo, "fstat failed", shard);
  }
  const FileIdentity& want = shard_files_[shard];
  if (!S_ISREG(status.st_mode) || status.st_nlink != 1 || status.st_dev != want.device ||
      status.st_ino != want.inode || Generation(fd.get()) != want.generation ||
      std::cmp_not_equal(status.st_size, want.size) ||
      status.st_ctim.tv_sec != want.changed_seconds ||
      status.st_ctim.tv_nsec != want.changed_nanoseconds) {
    return Fail(Rule::kFileType, "the shard is not the file validated at open", shard);
  }
  const int flags = ::fcntl(fd.get(), F_GETFL);
  if (flags < 0 || ::fcntl(fd.get(), F_SETFL, flags & ~O_NONBLOCK) != 0) {
    return Fail(Rule::kIo, "fcntl failed", shard);
  }
  return fd;
}

std::expected<std::string, Error> Artifact::ReadMetadata(std::string_view name) const {
  const std::string path = "meta/" + std::string(name);
  const auto listed = std::ranges::find_if(files_, [&path](const ListedFile& f) {
    return f.role == FileRole::kSourceMetadata && f.path == path;
  });
  if (name.empty() || name.contains('/') || listed == files_.end()) {
    return Fail(Rule::kFileSet, "no such kept metadata file", kNoItem);
  }
  const Item item = static_cast<Item>(listed - files_.begin());
  FileDescriptor meta(
      ::openat(root_.get(), "meta", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
  if (!meta.valid()) {
    return Fail(Rule::kFileType, "the meta directory is gone or replaced", item);
  }
  // The listed size bounds the read (the manifest's caps bound that).
  auto bytes = ReadDocument(meta.get(), std::string(name).c_str(), listed->bytes.value(),
                            kMaxMetadataBytes, item);
  if (!bytes) {
    return Pass(bytes);
  }
  if (Sha256Of(*bytes) != listed->sha256) {
    return Fail(Rule::kHash, "kept metadata differs from its listed digest", item);
  }
  return bytes;
}

}  // namespace llmp::artifact
