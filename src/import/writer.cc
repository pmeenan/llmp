// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "import/writer.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <format>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <vector>

#include "artifact/format_json.h"
#include "base/gguf.h"
#include "base/json.h"
#include "components/step.h"
#include "import/stream.h"
#include "platform/lock_file.h"
#include "platform/sockets.h"
#include "spec/stored.h"

namespace llmp::import {
namespace {
namespace fs = std::filesystem;
using FD = artifact::FileDescriptor;
constexpr std::uint64_t kShardTarget = std::uint64_t{4} << 30U;
constexpr std::uint64_t kMaxOutput = std::uint64_t{1} << 50U;
std::unexpected<Error> Bad(std::string_view reason, Rule rule = Rule::kIo) {
  return std::unexpected(Error{.rule = rule, .reason = reason});
}
std::string Q(std::string_view value) {
  std::string result;
  base::json::AppendQuoted(value, result);
  return result;
}
std::string N(std::uint64_t value) { return std::to_string(value); }
std::string A(const std::vector<std::string>& values) {
  std::string text = "[";
  for (const auto& value : values) {
    if (text.size() > 1) text += ',';
    text += value;
  }
  return text + ']';
}
std::string O(std::map<std::string, std::string> values) {
  std::string text = "{";
  for (const auto& [key, value] : values) {
    if (text.size() > 1) text += ',';
    text += Q(key) + ':' + value;
  }
  return text + '}';
}
std::string Hex(std::string_view bytes) {
  return base::ToHex(base::Sha256().Update(bytes).Finish());
}
std::uint64_t Align(std::uint64_t n, std::uint64_t alignment) {
  return (n + alignment - 1) / alignment * alignment;
}
bool SafeName(std::string_view name) {
  return !name.empty() && name.size() <= 255 &&
         std::ranges::all_of(name,
                             [](char c) {
                               return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                                      (c >= '0' && c <= '9') || c == '_' || c == '.' || c == '-';
                             }) &&
         name != "." && name != "..";
}
std::expected<void, Error> Write(int fd, std::uint64_t at, std::span<const std::byte> bytes) {
  while (!bytes.empty()) {
    const auto n = ::pwrite(fd, bytes.data(), bytes.size(), static_cast<off_t>(at));
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return Bad("cannot write an artifact file");
    at += static_cast<std::uint64_t>(n);
    bytes = bytes.subspan(static_cast<std::size_t>(n));
  }
  return {};
}
std::expected<FD, Error> Create(const fs::path& path) {
  FD file(::open(path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600));
  if (!file.valid()) return Bad("cannot create an artifact staging file");
  return file;
}
std::expected<void, Error> Save(const fs::path& path, std::string_view text) {
  auto file = Create(path);
  if (!file) return std::unexpected(file.error());
  if (auto ok = Write(file->get(), 0, std::as_bytes(std::span(text))); !ok) return ok;
  if (::fsync(file->get()) != 0) return Bad("cannot sync an artifact file");
  return {};
}
std::expected<void, Error> Sync(const fs::path& path) {
  FD file(::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  if (!file.valid() || ::fsync(file.get()) != 0) return Bad("cannot sync an artifact directory");
  return {};
}
struct Cleanup {
  fs::path path{};
  ~Cleanup() {
    std::error_code ignored;
    if (!path.empty()) fs::remove_all(path, ignored);
  }
};
struct Member {
  std::uint32_t tensor = 0;
  std::optional<std::uint32_t> expert{};
  std::uint64_t offset = 0, bytes = 0, readable = 0;
  format::TensorFormat format{};
};
struct Group {
  std::string kind{};
  std::optional<std::uint32_t> layer{}, expert{};
  std::uint32_t shard = 0;
  std::uint64_t offset = 0, used = 0, stored = 0, first_chunk = 0;
  std::vector<Member> members{};
};
struct Shard {
  std::string path{}, header{};
  std::uint64_t bytes = 0;
};
struct Copy {
  std::uint32_t file = 0, shard = 0;
  std::uint64_t source = 0, output = 0, bytes = 0, written = 0;
};
struct Layout {
  std::vector<Group> groups{};
  std::vector<Shard> shards{};
  std::vector<Copy> copies{};
  std::uint32_t experts = 0;
};
struct GroupKey {
  // Table, global, each layer followed by its experts, head.
  std::uint32_t order = 0, layer = 0;
  auto operator<=>(const GroupKey&) const = default;
};
std::expected<Layout, Error> Place(const NormalizedPlan& plan, std::uint64_t funding) {
  const auto& tensors = plan.canonical().tensors;
  if (tensors.empty() || tensors.size() > artifact::kMaxEntries ||
      tensors.size() != plan.routes().size())
    return Bad("invalid normalized output count", Rule::kBounds);
  std::uint64_t expanded = 0;
  for (const auto& t : tensors) {
    const auto count = t.experts ? t.format.shape.ne[2] : 1;
    if (count > artifact::kMaxEntries - expanded)
      return Bad("expert expansion exceeds the artifact entry cap", Rule::kBounds);
    expanded += count;
  }
  // Explicit conservative storage ceiling includes JSON serialization, layout,
  // maps, parsed-prefix buffers, stream buffers and the artifact reader.
  const std::uint64_t payload = plan.cost().payload_bytes;
  if (payload > kMaxOutput) return Bad("source payload exceeds output cap", Rule::kBounds);
  const std::uint64_t chunk_bound = payload / artifact::kChunkBytes + expanded * 2;
  if (chunk_bound > artifact::kMaxIndexBytes / 128)
    return Bad("output chunk metadata exceeds index cap", Rule::kBounds);
  const std::uint64_t bound = (std::uint64_t{256} << 20U) + expanded * 8192 + chunk_bound * 512 +
                              plan.source().files.size() * 1024;
  if (funding < bound) return Bad("artifact metadata has not been funded", Rule::kBounds);
  std::map<std::string, GroupKey, std::less<>> owners;
  const auto bind = [&](const spec::ComponentRef& component, GroupKey key) {
    for (const auto& w : component.weights) {
      auto [it, inserted] = owners.try_emplace(w.tensor, key);
      if (!inserted && it->second != key) {
        // A tied embedding remains a row table; other shared weights are global.
        if (key.order == 0 || it->second.order == 0)
          it->second = {0, 0};
        else
          it->second = {1, 0};
      }
    }
  };
  for (const auto& stage : plan.spec().stages) {
    if (stage.embedding) bind(*stage.embedding, {0, 0});
    if (stage.stream.mixing) bind(*stage.stream.mixing, {1, 0});
    for (const auto& side : stage.side) bind(side.source, {1, 0});
    for (std::size_t i = 0; i < stage.layers.size(); ++i)
      for (const auto& c : stage.layers[i].components) bind(c, {2, static_cast<std::uint32_t>(i)});
    if (stage.heads.final_norm) bind(*stage.heads.final_norm, {3, 0});
    if (stage.heads.output) bind(*stage.heads.output, {3, 0});
    for (const auto& c : stage.heads.mtp) bind(c, {3, 0});
  }
  std::map<GroupKey, std::vector<std::uint32_t>> batches;
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    if (!SafeName(tensors[i].name))
      return Bad("a tensor name is outside the artifact vocabulary", Rule::kUnsupported);
    const auto found = owners.find(tensors[i].name);
    if (found == owners.end()) return Bad("a normalized tensor is unbound", Rule::kConfig);
    batches[found->second].push_back(static_cast<std::uint32_t>(i));
  }
  Layout out;
  std::uint64_t total = 0, next_chunk = 0;
  for (auto& [key, indices] : batches) {
    std::ranges::sort(indices, [&](auto a, auto b) { return tensors[a].name < tensors[b].name; });
    std::vector<std::uint32_t> dense, experts;
    for (auto i : indices) (tensors[i].experts ? experts : dense).push_back(i);
    if (!experts.empty()) {
      if (key.order != 2) return Bad("an expert array has no unique layer", Rule::kUnsupported);
      const auto count = tensors[experts.front()].format.shape.ne[2];
      if (count == 0 || count > artifact::kMaxExperts || (out.experts && out.experts != count))
        return Bad("expert counts disagree", Rule::kConfig);
      out.experts = static_cast<std::uint32_t>(count);
    }
    const auto add = [&](const std::vector<std::uint32_t>& members,
                         std::optional<std::uint32_t> expert) -> std::expected<void, Error> {
      if (members.empty()) return {};
      Group g{.kind = expert           ? "expert"
                      : key.order == 0 ? "table"
                      : key.order == 1 ? "global"
                      : key.order == 2 ? "layer"
                                       : "head",
              .layer = key.order == 2 ? std::optional(key.layer) : std::nullopt,
              .expert = expert};
      for (auto i : members) {
        const auto& t = tensors[i];
        auto format = t.format;
        if (expert) {
          if (format.shape.rank != 3 || format.shape.ne[2] != out.experts)
            return Bad("an expert array has inconsistent geometry", Rule::kConfig);
          format.shape.rank = 2;
          format.shape.ne[2] = 1;
        }
        const auto parts = format::PartsOf(format);
        if (!parts) return Bad("an output format is invalid", Rule::kConfig);
        const auto part = parts->Find(t.part);
        if (!part) return Bad("an output format lacks its part", Rule::kConfig);
        if (part->readable > kMaxOutput - g.used - 256)
          return Bad("artifact output exceeds its byte cap", Rule::kBounds);
        const auto offset = Align(g.used, artifact::kMemberAlignment);
        g.members.push_back({i, expert, offset, part->bytes, part->readable, format});
        g.used = offset + part->readable;
      }
      g.stored = Align(g.used, artifact::kFileAlignment);
      if (g.stored > kMaxOutput - total)
        return Bad("artifact output exceeds its byte cap", Rule::kBounds);
      total += g.stored;
      if (out.shards.empty() ||
          (out.shards.back().bytes && out.shards.back().bytes + g.stored > kShardTarget)) {
        if (out.shards.size() >= 99999) return Bad("too many artifact shards", Rule::kBounds);
        out.shards.push_back({.path = std::format("data/{:05}.safetensors", out.shards.size())});
      }
      g.shard = static_cast<std::uint32_t>(out.shards.size() - 1);
      g.offset = out.shards.back().bytes;
      out.shards.back().bytes += g.stored;
      g.first_chunk = next_chunk;
      next_chunk += (g.stored + artifact::kChunkBytes - 1) / artifact::kChunkBytes;
      out.groups.push_back(std::move(g));
      return {};
    };
    if (auto ok = add(dense, std::nullopt); !ok) return std::unexpected(ok.error());
    if (!experts.empty())
      for (std::uint32_t e = 0; e < out.experts; ++e)
        if (auto ok = add(experts, e); !ok) return std::unexpected(ok.error());
  }
  if (out.groups.size() > artifact::kMaxEntries)
    return Bad("too many artifact groups", Rule::kBounds);
  // Shard headers have the exact byte order reconstructed by Artifact::Open.
  std::vector<std::uint64_t> pads(out.shards.size(), 0);
  for (auto& shard : out.shards)
    shard.header =
        std::string(8, '\0') + R"({"__metadata__":{"format":"llmp-shard","format_version":"1"})";
  const auto entry = [&](std::uint32_t s, std::string_view name, std::string_view dtype,
                         const std::vector<std::uint64_t>& shape, std::uint64_t begin,
                         std::uint64_t end) {
    std::vector<std::string> dims;
    for (auto n : shape) dims.push_back(N(n));
    out.shards[s].header += ',' + Q(name) + ":{\"dtype\":" + Q(dtype) + ",\"shape\":" + A(dims) +
                            ",\"data_offsets\":[" + N(begin) + ',' + N(end) + "]}";
  };
  for (const auto& g : out.groups) {
    auto at = g.offset;
    const auto pad = [&](std::uint64_t end) {
      if (at < end) entry(g.shard, "~pad." + N(pads[g.shard]++), "U8", {end - at}, at, end);
    };
    for (const auto& m : g.members) {
      const auto& t = tensors[m.tensor];
      const auto& route = plan.routes()[m.tensor];
      const auto target = g.offset + m.offset;
      pad(target);
      const auto view = artifact::ContainerViewOf(m.format, t.part, m.bytes);
      entry(g.shard, t.name + (m.expert ? "#" + N(*m.expert) : ""), view.dtype, view.shape, target,
            target + m.bytes);
      at = target + m.bytes;
      const auto input_offset = m.expert ? *m.expert * m.bytes : 0;
      if (input_offset > route.extent.bytes || m.bytes > route.extent.bytes - input_offset)
        return Bad("an output slice is outside its copy route", Rule::kConfig);
      auto cursor = plan.CursorFor(m.tensor);
      if (!cursor) return std::unexpected(cursor.error());
      if (input_offset != 0) {
        auto skipped = cursor->Next(input_offset);
        if (!skipped || !*skipped || (**skipped).bytes != input_offset)
          return Bad("an expert slice cannot advance its source cursor", Rule::kConfig);
      }
      auto copy = cursor->Next(m.bytes);
      if (!copy || !*copy || (**copy).bytes != m.bytes || (**copy).output_offset != input_offset)
        return Bad("an output slice does not match its source cursor", Rule::kConfig);
      out.copies.push_back(
          {(**copy).file, g.shard, (**copy).source_offset, target, (**copy).bytes, 0});
    }
    pad(g.offset + g.stored);
  }
  for (auto& shard : out.shards) {
    shard.header += '}';
    if (shard.header.size() > artifact::kMaxShardHeader - 4096)
      return Bad("shard header exceeds cap", Rule::kBounds);
    shard.header.resize(Align(shard.header.size(), 4096), ' ');
    const auto size = shard.header.size() - 8;
    for (std::size_t i = 0; i < 8; ++i)
      shard.header[i] = static_cast<char>((size >> (8U * i)) & 255U);
  }
  std::ranges::sort(out.copies, {},
                    [](const Copy& c) { return std::tuple(c.file, c.source, c.shard, c.output); });
  return out;
}
std::string FileJson(std::string_view path, std::string_view role, std::uint64_t bytes,
                     std::string_view hash) {
  return O({{"bytes", N(bytes)}, {"path", Q(path)}, {"role", Q(role)}, {"sha256", Q(hash)}});
}
// Deep verification is deliberately payload bounded. Structural verification
// already reconstructs headers and validates the zero-pad geometry.
std::expected<void, Error> Deep(const fs::path& root, const artifact::Artifact& artifact) {
  std::vector<std::byte> buffer(artifact::kChunkBytes);
  for (const auto& listed : artifact.files()) {
    auto input = OpenCheckpointFile(root / listed.path, kNoItem);
    if (!input) return std::unexpected(input.error());
    base::Sha256 hash;
    for (std::uint64_t at = 0; at < listed.bytes.value();) {
      auto bytes = std::span(buffer).first(static_cast<std::size_t>(
          std::min<std::uint64_t>(buffer.size(), listed.bytes.value() - at)));
      if (auto ok = ReadExactly(input->fd.get(), at, bytes, kNoItem); !ok) return ok;
      hash.Update(bytes);
      at += bytes.size();
    }
    if (input->bytes != listed.bytes.value() || hash.Finish() != listed.sha256)
      return Bad("a staged or existing artifact file failed full verification");
  }
  for (std::size_t gi = 0; gi < artifact.groups().size(); ++gi) {
    const auto& g = artifact.groups()[gi];
    auto input = OpenCheckpointFile(root / artifact.shards()[g.shard].path, kNoItem);
    if (!input) return std::unexpected(input.error());
    for (std::uint32_t k = 0; k < g.chunks; ++k) {
      auto range = artifact::ChunkRangeOf(artifact.layout(), {static_cast<std::uint32_t>(gi), k});
      if (!range) return Bad("an artifact chunk has invalid geometry");
      auto bytes = std::span(buffer).first(static_cast<std::size_t>(range->length.value()));
      if (auto ok = ReadExactly(input->fd.get(), range->file_offset.value(), bytes, kNoItem); !ok)
        return ok;
      if (base::Sha256().Update(bytes).Finish() != artifact.chunk_sha256()[g.first_chunk + k])
        return Bad("an artifact chunk failed verification");
    }
  }
  return {};
}
}  // namespace

std::expected<WriteResult, Error> WriteArtifact(const NormalizedPlan& plan,
                                                const WriteOptions& options) {
  const auto& source = plan.source();
  if (options.store.empty() || !SafeName(options.build) ||
      (!options.expected.empty() && options.expected.size() != source.files.size()))
    return Bad("invalid writer options", Rule::kConfig);
  auto layout = Place(plan, options.metadata_bytes);
  if (!layout) return std::unexpected(layout.error());
  std::set<std::string_view> source_names;
  for (const auto& file : source.files) {
    if (!SafeName(file.name) || !source_names.insert(file.name).second || file.path.empty() ||
        file.parsed.bytes == 0 || file.parsed.bytes > file.bytes)
      return Bad("a parsed source lacks a valid identity receipt", Rule::kConfig);
  }
  std::error_code error;
  if (auto ok = CreateImportDirectory(options.store, true); !ok) return std::unexpected(ok.error());
  const fs::path staging = options.store / ".staging";
  auto status = fs::symlink_status(staging, error);
  if (error && error != std::errc::no_such_file_or_directory)
    return Bad("cannot examine staging directory");
  if (fs::is_symlink(status)) return Bad("artifact staging must not be a symlink");
  error.clear();
  if (auto ok = CreateImportDirectory(staging, false); !ok) return std::unexpected(ok.error());
  auto lock = platform::LockFile::Acquire(staging / "native-import.lock");
  if (!lock) return Bad("another import owns this store or its lock cannot be opened");
  std::array<std::byte, 32> random{};
  if (!platform::FillRandom(random)) return Bad("cannot generate a private staging name");
  const fs::path job = staging / ("job-" + base::ToHex(base::Sha256().Update(random).Finish()));
  if (!fs::create_directory(job, error) || error)
    return Bad("cannot create private staging directory");
  Cleanup cleanup{job};
  if (::chmod(job.c_str(), 0700) != 0) return Bad("cannot protect staging directory");
  fs::create_directory(job / "data", error);
  if (error) return Bad("cannot create shard directory");
  fs::create_directory(job / "meta", error);
  if (error) return Bad("cannot create metadata directory");
  for (const auto& shard : layout->shards) {
    auto file = Create(job / shard.path);
    if (!file) return std::unexpected(file.error());
    if (auto ok = Write(file->get(), 0, std::as_bytes(std::span(shard.header))); !ok)
      return std::unexpected(ok.error());
    if (::ftruncate(file->get(), static_cast<off_t>(shard.header.size() + shard.bytes)) != 0)
      return Bad("cannot size a staging shard");
  }
  std::map<std::string, std::string> files;
  std::map<std::string, std::string> sources;
  std::uint64_t kept_bytes = 0;
  std::size_t next_copy = 0;
  FD output;
  std::optional<std::uint32_t> output_shard;
  for (std::size_t f = 0; f < source.files.size(); ++f) {
    const auto& input = source.files[f];
    const bool gguf = source.container == "gguf";
    const bool keep = gguf || (input.name.ends_with(".json") &&
                               ResolvesInside(input.path.parent_path(), input.path));
    // Source metadata copies must stay within the checkpoint (cache blobs
    // count); ordinary parsed weight files retain their usual symlink rules.
    const auto prefix_limit =
        gguf ? std::min(input.parsed.bytes, artifact::kMaxMetadataBytes) : input.bytes;
    if (keep && prefix_limit > artifact::kMaxMetadataBytes)
      return Bad("kept metadata exceeds its artifact cap", Rule::kBounds);
    std::string metadata;
    if (keep) metadata.reserve(static_cast<std::size_t>(prefix_limit));
    const auto streamed = StreamSource(
        input, f,
        [&](std::uint64_t at, std::span<const std::byte> bytes) -> std::expected<void, Error> {
          if (keep && at < prefix_limit) {
            const auto count =
                static_cast<std::size_t>(std::min<std::uint64_t>(bytes.size(), prefix_limit - at));
            metadata.append(reinterpret_cast<const char*>(bytes.data()), count);
          }
          const auto end = at + bytes.size();
          for (std::size_t i = next_copy; i < layout->copies.size(); ++i) {
            auto& c = layout->copies[i];
            if (c.file != f || c.source >= end) break;
            const auto first = std::max(c.source, at);
            const auto last = std::min(c.source + c.bytes, end);
            if (first < last) {
              if (!output_shard || *output_shard != c.shard) {
                output =
                    FD(::open((job / layout->shards[c.shard].path).c_str(), O_RDWR | O_CLOEXEC));
                if (!output.valid()) return Bad("cannot reopen a staging shard");
                output_shard = c.shard;
              }
              if (first - c.source != c.written) return Bad("copy progress is not contiguous");
              const auto piece = bytes.subspan(static_cast<std::size_t>(first - at),
                                               static_cast<std::size_t>(last - first));
              if (auto ok =
                      Write(output.get(),
                            layout->shards[c.shard].header.size() + c.output + c.written, piece);
                  !ok)
                return ok;
              c.written += piece.size();
            }
            if (c.written == c.bytes) next_copy = i + 1;
          }
          return {};
        },
        options.expected.empty() ? std::nullopt : options.expected[f]);
    if (!streamed) return std::unexpected(streamed.error());
    sources.emplace(input.name, O({{"bytes", N(input.bytes)},
                                   {"name", Q(input.name)},
                                   {"sha256", Q(base::ToHex(*streamed))}}));
    if (keep) {
      std::string name = input.name;
      if (gguf) {
        auto header = base::gguf::ParseMetadata(std::as_bytes(std::span(metadata)));
        if (!header || header->metadata_end > artifact::kMaxMetadataBytes)
          return Bad("GGUF metadata cannot be kept within artifact bounds", Rule::kBounds);
        metadata.resize(static_cast<std::size_t>(header->metadata_end));
        std::fill(metadata.begin() + 8, metadata.begin() + 16, '\0');
        metadata[4] = '\3';  // Kept metadata uses GGUF v3; v2 has the same parsed fields.
        name = input.name.substr(0, input.name.size() - 5) + ".kv.gguf";
      }
      if (metadata.size() > artifact::kMaxMetadataTotal - kept_bytes)
        return Bad("kept metadata total exceeds artifact cap", Rule::kBounds);
      kept_bytes += metadata.size();
      const std::string path = "meta/" + name;
      if (auto ok = Save(job / path, metadata); !ok) return std::unexpected(ok.error());
      files.emplace(path, FileJson(path, "source-metadata", metadata.size(), Hex(metadata)));
    }
  }
  output = FD();
  for (const auto& copy : layout->copies)
    if (copy.written != copy.bytes) return Bad("a source stream left an output range unwritten");

  std::vector<std::string> chunks, groups, resources, arrays, shards, transformations;
  std::vector<std::byte> buffer(artifact::kChunkBytes);
  for (const auto& shard : layout->shards) {
    FD file(::open((job / shard.path).c_str(), O_RDWR | O_CLOEXEC));
    if (!file.valid()) return Bad("cannot reopen a written shard");
    base::Sha256 hash;
    const auto size = shard.header.size() + shard.bytes;
    for (std::uint64_t at = 0; at < size;) {
      auto bytes = std::span(buffer).first(
          static_cast<std::size_t>(std::min<std::uint64_t>(buffer.size(), size - at)));
      if (auto ok = ReadExactly(file.get(), at, bytes, kNoItem); !ok)
        return std::unexpected(ok.error());
      hash.Update(bytes);
      at += bytes.size();
    }
    if (::fsync(file.get()) != 0) return Bad("cannot sync a written shard");
    files.emplace(shard.path, FileJson(shard.path, "shard", size, base::ToHex(hash.Finish())));
    shards.push_back(O({{"data_bytes", N(shard.bytes)},
                        {"data_offset", N(shard.header.size())},
                        {"header_sha256", Q(Hex(shard.header))},
                        {"path", Q(shard.path)}}));
  }
  for (std::size_t gi = 0; gi < layout->groups.size(); ++gi) {
    const auto& g = layout->groups[gi];
    const auto& shard = layout->shards[g.shard];
    auto file = OpenCheckpointFile(job / shard.path, kNoItem);
    if (!file) return std::unexpected(file.error());
    for (std::uint64_t at = 0; at < g.stored; at += artifact::kChunkBytes) {
      auto bytes = std::span(buffer).first(
          static_cast<std::size_t>(std::min<std::uint64_t>(buffer.size(), g.stored - at)));
      if (auto ok =
              ReadExactly(file->fd.get(), shard.header.size() + g.offset + at, bytes, kNoItem);
          !ok)
        return std::unexpected(ok.error());
      chunks.push_back(Q(base::ToHex(base::Sha256().Update(bytes).Finish())));
    }
    groups.push_back(O({{"expert", g.expert ? N(*g.expert) : "null"},
                        {"first_chunk", N(g.first_chunk)},
                        {"kind", Q(g.kind)},
                        {"layer", g.layer ? N(*g.layer) : "null"},
                        {"offset", N(g.offset)},
                        {"shard", N(g.shard)},
                        {"stored_bytes", N(g.stored)},
                        {"used_bytes", N(g.used)}}));
    for (const auto& m : g.members) {
      const auto& tensor = plan.canonical().tensors[m.tensor];
      std::string format;
      artifact::AppendFormatJson(m.format, format);
      if (!m.expert) {
        std::map<std::string, std::string> r{{"bytes", N(m.bytes)},
                                             {"format", format},
                                             {"group", N(gi)},
                                             {"name", Q(tensor.name)},
                                             {"offset", N(m.offset)},
                                             {"part", Q(format::PartRoleName(tensor.part))},
                                             {"readable_bytes", N(m.readable)},
                                             {"roles", A({Q(tensor.name)})}};
        if (g.kind == "table" && format::RowBytes(m.format, tensor.part))
          r.emplace("access", Q("rows"));
        resources.push_back(O(std::move(r)));
      } else if (*m.expert == 0) {
        arrays.push_back(O({{"count", N(layout->experts)},
                            {"first_group", N(gi)},
                            {"format", format},
                            {"group_offset", N(m.offset)},
                            {"layer", N(*g.layer)},
                            {"name", Q(tensor.name)},
                            {"part", Q(format::PartRoleName(tensor.part))},
                            {"readable_bytes", N(m.readable)},
                            {"slice_bytes", N(m.bytes)}}));
        transformations.push_back(O({{"count", N(layout->experts)},
                                     {"kind", Q("expert-slice")},
                                     {"tensor", Q(tensor.name)}}));
      }
    }
  }
  const std::string index = O({{"chunk_bytes", N(artifact::kChunkBytes)},
                               {"chunk_sha256", A(chunks)},
                               {"expert_arrays", A(arrays)},
                               {"file_alignment", N(artifact::kFileAlignment)},
                               {"format", Q("llmp-index")},
                               {"format_version", "1"},
                               {"groups", A(groups)},
                               {"member_alignment", N(artifact::kMemberAlignment)},
                               {"resources", A(resources)},
                               {"shards", A(shards)}}) +
                            '\n';
  const auto spec = spec::ToJson(plan.spec());
  if (index.size() > artifact::kMaxIndexBytes || spec.size() > artifact::kMaxSpecBytes)
    return Bad("an artifact document exceeds its cap", Rule::kBounds);
  if (auto ok = Save(job / "index.json", index); !ok) return std::unexpected(ok.error());
  if (auto ok = Save(job / "spec.json", spec); !ok) return std::unexpected(ok.error());
  files.emplace("index.json", FileJson("index.json", "index", index.size(), Hex(index)));
  files.emplace("spec.json", FileJson("spec.json", "spec", spec.size(), Hex(spec)));
  std::vector<std::string> file_list, source_list;
  for (const auto& [path, json] : files) file_list.push_back(json);
  for (const auto& [path, json] : sources) source_list.push_back(json);
  std::ranges::sort(transformations);
  const auto manifest = O({{"experimental", "true"},
                           {"files", A(file_list)},
                           {"format", Q("llmp-artifact")},
                           {"format_version", "1"},
                           {"importer", O({{"build", Q(options.build)},
                                           {"name", Q("llmp-import")},
                                           {"options_sha256", Q(Hex(""))},
                                           {"version", Q("1")}})},
                           {"layout", O({{"byte_order", Q("little")},
                                         {"chunk_bytes", N(artifact::kChunkBytes)},
                                         {"container", Q("safetensors")},
                                         {"file_alignment", N(artifact::kFileAlignment)},
                                         {"member_alignment", N(artifact::kMemberAlignment)},
                                         {"profile", Q("spark-v0")},
                                         {"profile_version", "0"}})},
                           {"model", O({{"architecture", Q(source.architecture)},
                                        {"container", Q(source.container)},
                                        {"expert_count", N(layout->experts)}})},
                           {"source", A(source_list)},
                           {"transformations", A(transformations)}}) +
                        '\n';
  if (manifest.size() > artifact::kMaxManifestBytes)
    return Bad("artifact manifest exceeds its cap", Rule::kBounds);
  if (auto ok = Save(job / "manifest.json", manifest); !ok) return std::unexpected(ok.error());
  const std::string id = Hex(manifest);
  auto opened = artifact::Artifact::Open(job, {.expected_id = id});
  if (!opened) return Bad("the produced artifact failed structural verification", Rule::kConfig);
  auto stored = spec::StoredSpec(*opened);
  if (!stored || *stored != plan.spec())
    return Bad("the produced artifact has an invalid embedded spec", Rule::kConfig);
  auto stored_source = spec::SourceOf(*opened);
  if (!stored_source || !components::ValidateSpec(*stored, stored_source->tensors))
    return Bad("the produced artifact does not bind its embedded spec", Rule::kConfig);
  if (auto ok = Deep(job, *opened); !ok) return std::unexpected(ok.error());
  for (const auto& directory : {job / "data", job / "meta", job})
    if (auto ok = Sync(directory); !ok) return std::unexpected(ok.error());
  const auto destination = options.store / id;
  const bool exists = fs::exists(destination, error);
  if (error) return Bad("cannot inspect an existing artifact");
  if (exists) {
    auto previous = artifact::Artifact::Open(destination);
    if (!previous) return Bad("the existing artifact is damaged");
    if (auto ok = Deep(destination, *previous); !ok) return std::unexpected(ok.error());
    return WriteResult{id, destination, true};
  }
  fs::rename(job, destination, error);
  if (error) return Bad("cannot publish artifact");
  cleanup.path.clear();
  if (auto ok = Sync(staging); !ok)
    return Bad("artifact published; staging directory durability sync failed");
  if (auto ok = Sync(options.store); !ok)
    return Bad("artifact published; store directory durability sync failed");
  return WriteResult{id, destination, false};
}
}  // namespace llmp::import
