// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "artifact_writer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

#include "artifact/format_json.h"
#include "artifact/layout.h"
#include "base/check.h"
#include "base/sha256.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"

namespace llmp::test {

namespace fs = std::filesystem;

namespace {

constexpr std::uint64_t kFile = artifact::kFileAlignment;
constexpr std::uint64_t kChunk = artifact::kChunkBytes;
constexpr std::uint64_t kMember = artifact::kMemberAlignment;
constexpr std::string_view kShardMetadata =
    R"({"__metadata__":{"format":"llmp-shard","format_version":"1"})";

std::uint64_t AlignUp(std::uint64_t n, std::uint64_t unit) { return (n + unit - 1) / unit * unit; }

std::string Hex(std::string_view bytes) {
  return base::ToHex(base::Sha256().Update(bytes).Finish());
}

std::string_view KindName(artifact::GroupKind kind) {
  switch (kind) {
    case artifact::GroupKind::kTable:
      return "table";
    case artifact::GroupKind::kLayer:
      return "layer";
    case artifact::GroupKind::kExpert:
      return "expert";
    case artifact::GroupKind::kGlobal:
      return "global";
    case artifact::GroupKind::kHead:
      return "head";
  }
  return "layer";
}

format::PartExtent ExtentOf(const format::TensorFormat& f, format::PartRole part) {
  const auto parts = format::PartsOf(f);
  base::Check(parts.has_value(), "a writer plan's formats are valid");
  const auto extent = parts->Find(part);
  base::Check(extent.has_value(), "a writer plan's parts are its formats'");
  return *extent;
}

// A placed member of a group: where, how big, and what fills it.
struct Placed {
  std::string name;  // the container entry's
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
  std::uint64_t readable = 0;
  artifact::ContainerView view;
};

struct PlacedGroup {
  const V1Group* plan = nullptr;
  std::optional<std::uint32_t> expert;
  std::uint32_t shard = 0;
  std::uint64_t offset = 0;
  std::uint64_t used = 0;
  std::uint64_t stored = 0;
  std::uint64_t first_chunk = 0;
  std::vector<Placed> members;
};

void Fill(std::string& data, std::uint64_t at, std::uint64_t bytes, std::string_view name) {
  std::uint8_t salt = 0;
  for (const char c : name) {
    salt = static_cast<std::uint8_t>((salt * 31U) + static_cast<std::uint8_t>(c));
  }
  for (std::uint64_t i = 0; i < bytes; ++i) {
    data[at + i] = static_cast<char>(static_cast<std::uint8_t>(salt + (i * 7U) + 1U));
  }
}

Json SortedTransformations(std::vector<Json> list) {
  std::ranges::sort(list, {}, [](const Json& j) {
    std::string out;
    j.Write(out);
    return out;
  });
  return Json(Json::Array(std::move(list)));
}

void WriteFile(const fs::path& path, std::string_view data) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(data.data(), static_cast<std::streamsize>(data.size()));
}

Json FileEntry(std::string_view path, std::string_view role, std::string_view bytes) {
  return Json(Json::Object{
      {"bytes", bytes.size()}, {"path", path}, {"role", role}, {"sha256", Hex(bytes)}});
}

Json Importer(std::string_view name, std::string_view version, std::string_view build) {
  return Json(Json::Object{
      {"build", build}, {"name", name}, {"options_sha256", Hex("")}, {"version", version}});
}

}  // namespace

// ------------------------------------------------------------------ Json

Json& Json::operator[](std::string_view key) {
  if (!std::holds_alternative<Object>(v_)) {
    v_ = Object{};
  }
  auto& o = std::get<Object>(v_);
  const auto found = o.find(key);
  if (found != o.end()) {
    return found->second;
  }
  return o.emplace(std::string(key), Json()).first->second;
}

Json& Json::at(std::size_t i) { return std::get<Array>(v_).at(i); }
Json::Array& Json::array() { return std::get<Array>(v_); }
Json::Object& Json::object() { return std::get<Object>(v_); }

void Json::Erase(std::string_view key) {
  auto& o = std::get<Object>(v_);
  const auto found = o.find(key);
  if (found != o.end()) {
    o.erase(found);
  }
}

void Json::Write(std::string& out) const {
  if (std::holds_alternative<std::nullptr_t>(v_)) {
    out += "null";
  } else if (const bool* b = std::get_if<bool>(&v_)) {
    out += *b ? "true" : "false";
  } else if (const std::int64_t* n = std::get_if<std::int64_t>(&v_)) {
    out += std::to_string(*n);
  } else if (const std::string* s = std::get_if<std::string>(&v_)) {
    out += '"';
    out += *s;
    out += '"';
  } else if (const Array* a = std::get_if<Array>(&v_)) {
    out += '[';
    for (std::size_t i = 0; i < a->size(); ++i) {
      out += i > 0 ? "," : "";
      (*a)[i].Write(out);
    }
    out += ']';
  } else if (const Object* o = std::get_if<Object>(&v_)) {
    out += '{';
    bool first = true;
    for (const auto& [k, v] : *o) {
      out += first ? "\"" : ",\"";
      first = false;
      out += k;
      out += "\":";
      v.Write(out);
    }
    out += '}';
  } else {
    out += std::get<Raw>(v_).text;
  }
}

std::string Json::Text() const {
  std::string out;
  Write(out);
  out += '\n';
  return out;
}

Json FormatJson(const format::TensorFormat& f) {
  std::string text;
  artifact::AppendFormatJson(f, text);
  return Json(Json::Raw{.text = std::move(text)});
}

// --------------------------------------------------------------- shards

std::string ShardHeader(std::string_view metadata, const std::vector<ShardEntry>& entries) {
  std::string text(8, '\0');
  text += metadata;
  for (const ShardEntry& e : entries) {
    text += ",\"";
    text += e.name;
    text += R"(":{"dtype":")";
    text += e.dtype;
    text += R"(","shape":[)";
    for (std::size_t k = 0; k < e.shape.size(); ++k) {
      text += k > 0 ? "," : "";
      text += std::to_string(e.shape[k]);
    }
    text += R"(],"data_offsets":[)";
    text += std::to_string(e.start);
    text += ',';
    text += std::to_string(e.end);
    text += "]}";
  }
  text += '}';
  const std::uint64_t data_offset = AlignUp(text.size(), kFile);
  const std::uint64_t length = data_offset - 8;
  for (std::size_t k = 0; k < 8; ++k) {
    text[k] = static_cast<char>((length >> (8U * k)) & 0xFFU);
  }
  text.resize(data_offset, ' ');
  return text;
}

// ------------------------------------------------------------- artifact

V1Files Render(const V1Plan& plan) {
  // Place every group's members, expanding an expert layer into its groups.
  std::vector<PlacedGroup> groups;
  std::uint32_t shard = 0;
  std::vector<std::uint64_t> shard_bytes = {0};
  for (const V1Group& g : plan.groups) {
    if (g.new_shard && !groups.empty()) {
      ++shard;
      shard_bytes.push_back(0);
    }
    std::vector<Placed> members;
    std::uint64_t end = 0;
    const auto place = [&](std::string_view name, const format::TensorFormat& f,
                           format::PartRole part, std::optional<std::uint64_t> at) {
      const format::PartExtent extent = ExtentOf(f, part);
      const std::uint64_t offset = at.value_or(AlignUp(end, kMember));
      members.push_back({.name = std::string(name),
                         .offset = offset,
                         .bytes = extent.bytes,
                         .readable = extent.readable,
                         .view = artifact::ContainerViewOf(f, part, extent.bytes)});
      end = std::max(end, offset + extent.readable);
    };
    for (const V1Resource& r : g.resources) {
      place(r.name, r.format, r.part, r.offset);
    }
    for (const V1Array& a : g.arrays) {
      place(a.name, a.format, a.part, a.offset);
    }
    std::ranges::sort(members, {}, &Placed::offset);
    const std::uint32_t copies = g.arrays.empty() ? 1 : plan.expert_count;
    for (std::uint32_t e = 0; e < copies; ++e) {
      PlacedGroup placed{.plan = &g,
                         .expert = g.arrays.empty() ? std::nullopt : std::optional(e),
                         .shard = shard,
                         .offset = shard_bytes[shard],
                         .used = end,
                         .stored = AlignUp(end, kFile),
                         .first_chunk = 0,
                         .members = members};
      if (!g.arrays.empty()) {
        for (Placed& m : placed.members) {
          m.name += "#" + std::to_string(e);
        }
      }
      shard_bytes[shard] += placed.stored;
      groups.push_back(std::move(placed));
    }
  }

  // The data, chunk digests and headers.
  V1Files files;
  std::vector<std::string> data(shard_bytes.size());
  for (std::size_t s = 0; s < data.size(); ++s) {
    data[s].assign(shard_bytes[s], '\0');
  }
  std::vector<std::vector<ShardEntry>> entries(shard_bytes.size());
  std::vector<std::uint64_t> pads(shard_bytes.size(), 0);
  Json::Array chunks;
  std::uint64_t next_chunk = 0;
  for (PlacedGroup& g : groups) {
    g.first_chunk = next_chunk;
    std::uint64_t cursor = g.offset;
    auto& list = entries[g.shard];
    const auto pad = [&](std::uint64_t from, std::uint64_t to) {
      list.push_back({.name = "~pad." + std::to_string(pads[g.shard]++),
                      .dtype = "U8",
                      .shape = {to - from},
                      .start = from,
                      .end = to});
    };
    for (const Placed& m : g.members) {
      const std::uint64_t at = g.offset + m.offset;
      if (at > cursor) {
        pad(cursor, at);
      }
      list.push_back({.name = m.name,
                      .dtype = std::string(m.view.dtype),
                      .shape = m.view.shape,
                      .start = at,
                      .end = at + m.bytes});
      Fill(data[g.shard], at, m.bytes, m.name);
      cursor = at + m.bytes;
    }
    if (g.offset + g.stored > cursor) {
      pad(cursor, g.offset + g.stored);
    }
    for (std::uint64_t k = 0; k * kChunk < g.stored; ++k) {
      const std::uint64_t length = std::min(kChunk, g.stored - (k * kChunk));
      chunks.emplace_back(
          Hex(std::string_view(data[g.shard]).substr(g.offset + (k * kChunk), length)));
      ++next_chunk;
    }
  }
  Json::Array shards;
  for (std::size_t s = 0; s < data.size(); ++s) {
    const std::string header = ShardHeader(kShardMetadata, entries[s]);
    std::string path = std::to_string(s);
    path.insert(0, 5 - path.size(), '0');
    shards.emplace_back(Json::Object{{"data_bytes", data[s].size()},
                                     {"data_offset", header.size()},
                                     {"header_sha256", Hex(header)},
                                     {"path", "data/" + path + ".safetensors"}});
    files.shards.push_back(header + data[s]);
  }

  // The index's groups, resources and expert arrays.
  Json::Array group_list;
  using Place = std::pair<std::uint64_t, std::uint64_t>;
  std::vector<std::pair<Place, Json>> resources;
  std::vector<std::pair<Place, Json>> arrays;
  std::vector<Json> transformations;
  for (std::size_t gid = 0; gid < groups.size(); ++gid) {
    const PlacedGroup& g = groups[gid];
    const V1Group& p = *g.plan;
    group_list.emplace_back(Json::Object{
        {"expert", g.expert ? Json(*g.expert) : Json()},
        {"first_chunk", g.first_chunk},
        {"kind", KindName(p.kind)},
        {"layer", p.layer ? Json(*p.layer) : Json()},
        {"offset", g.offset},
        {"shard", g.shard},
        {"stored_bytes", g.stored},
        {"used_bytes", g.used},
    });
    for (const V1Resource& r : p.resources) {
      const auto placed = std::ranges::find(g.members, r.name, &Placed::name);
      Json::Array roles = {Json(r.name)};
      for (const std::string& alias : r.aliases) {
        roles.emplace_back(alias);
        const std::string_view bytes =
            std::string_view(data[g.shard]).substr(g.offset + placed->offset, placed->bytes);
        transformations.emplace_back(Json::Object{{"kind", "dedupe-identical"},
                                                  {"resource", r.name},
                                                  {"role", alias},
                                                  {"sha256", Hex(bytes)}});
      }
      Json entry(Json::Object{{"bytes", placed->bytes},
                              {"format", FormatJson(r.format)},
                              {"group", gid},
                              {"name", r.name},
                              {"offset", placed->offset},
                              {"part", format::PartRoleName(r.part)},
                              {"readable_bytes", placed->readable},
                              {"roles", Json(std::move(roles))}});
      if (r.rows) {
        entry["access"] = "rows";
      }
      resources.emplace_back(Place(gid, placed->offset), std::move(entry));
    }
    if (!p.arrays.empty() && g.expert == 0U) {
      for (const V1Array& a : p.arrays) {
        const auto placed = std::ranges::find(g.members, a.name + "#0", &Placed::name);
        arrays.emplace_back(Place(gid, placed->offset),
                            Json::Object{{"count", plan.expert_count},
                                         {"first_group", gid},
                                         {"format", FormatJson(a.format)},
                                         {"group_offset", placed->offset},
                                         {"layer", p.layer.value_or(0)},
                                         {"name", a.name},
                                         {"part", format::PartRoleName(a.part)},
                                         {"readable_bytes", placed->readable},
                                         {"slice_bytes", placed->bytes}});
        transformations.emplace_back(Json::Object{
            {"count", plan.expert_count}, {"kind", "expert-slice"}, {"tensor", a.name}});
      }
    }
  }
  // Resources by (group, offset) and arrays by (first group, offset).
  std::ranges::stable_sort(resources, {}, &std::pair<Place, Json>::first);
  std::ranges::stable_sort(arrays, {}, &std::pair<Place, Json>::first);
  Json::Array resource_list;
  for (auto& [place, entry] : resources) {
    resource_list.push_back(std::move(entry));
  }
  Json::Array array_list;
  for (auto& [place, entry] : arrays) {
    array_list.push_back(std::move(entry));
  }
  files.index = Json(Json::Object{{"chunk_bytes", kChunk},
                                  {"chunk_sha256", Json(std::move(chunks))},
                                  {"expert_arrays", Json(std::move(array_list))},
                                  {"file_alignment", kFile},
                                  {"format", "llmp-index"},
                                  {"format_version", 1},
                                  {"groups", Json(std::move(group_list))},
                                  {"member_alignment", kMember},
                                  {"resources", Json(std::move(resource_list))},
                                  {"shards", Json(std::move(shards))}});

  // The manifest.
  for (const auto& [fold, tensors] : plan.folds) {
    Json::Array list(tensors.begin(), tensors.end());
    transformations.emplace_back(
        Json::Object{{"fold", fold}, {"kind", "fold"}, {"tensors", Json(std::move(list))}});
  }
  for (const auto& [from, tensors] : plan.relayouts) {
    Json::Array list(tensors.begin(), tensors.end());
    transformations.emplace_back(Json::Object{{"from", format::Info(from).name},
                                              {"kind", "relayout"},
                                              {"tensors", Json(std::move(list))}});
  }
  std::vector<std::pair<std::string, std::string>> sources = plan.sources;
  for (const auto& m : plan.meta) {
    sources.push_back(m);
  }
  std::ranges::sort(sources);
  Json::Array source_list;
  for (const auto& [name, bytes] : sources) {
    source_list.emplace_back(
        Json::Object{{"bytes", bytes.size()}, {"name", name}, {"sha256", Hex(bytes)}});
  }
  files.manifest = Json(Json::Object{
      {"experimental", true},
      {"files", Json(Json::Array{})},
      {"format", "llmp-artifact"},
      {"format_version", 1},
      {"importer", Importer(plan.importer, plan.importer_version, plan.importer_build)},
      {"layout", Json(Json::Object{{"byte_order", "little"},
                                   {"chunk_bytes", kChunk},
                                   {"container", "safetensors"},
                                   {"file_alignment", kFile},
                                   {"member_alignment", kMember},
                                   {"profile", "spark"},
                                   {"profile_version", 1}})},
      {"model", Json(Json::Object{{"architecture", plan.architecture},
                                  {"container", plan.container},
                                  {"expert_count", plan.expert_count}})},
      {"source", Json(std::move(source_list))},
      {"transformations", SortedTransformations(std::move(transformations))},
  });
  files.spec = plan.spec;
  files.meta = plan.meta;
  return files;
}

fs::path Publish(const fs::path& store, const V1Files& files, const PublishHooks& hooks) {
  std::string index = files.index.Text();
  if (hooks.index) {
    hooks.index(index);
  }
  Json manifest = files.manifest;
  if (!files.keep_files) {
    std::vector<std::pair<std::string, Json>> list;
    list.emplace_back("index.json", FileEntry("index.json", "index", index));
    for (std::size_t s = 0; s < files.shards.size(); ++s) {
      std::string path = std::to_string(s);
      path.insert(0, 5 - path.size(), '0');
      path = "data/" + path + ".safetensors";
      list.emplace_back(path, FileEntry(path, "shard", files.shards[s]));
    }
    for (const auto& [name, bytes] : files.meta) {
      list.emplace_back("meta/" + name, FileEntry("meta/" + name, "source-metadata", bytes));
    }
    if (files.spec) {
      list.emplace_back("spec.json", FileEntry("spec.json", "spec", *files.spec));
    }
    std::ranges::sort(list, {}, &std::pair<std::string, Json>::first);
    Json::Array entries;
    for (auto& [path, entry] : list) {
      entries.push_back(std::move(entry));
    }
    manifest["files"] = Json(std::move(entries));
  }
  std::string text = manifest.Text();
  if (hooks.manifest) {
    hooks.manifest(text);
  }
  const fs::path dir = store / Hex(text);
  std::error_code error;
  fs::remove_all(dir, error);
  fs::create_directories(dir, error);
  WriteFile(dir / "manifest.json", text);
  WriteFile(dir / "index.json", index);
  if (files.spec) {
    WriteFile(dir / "spec.json", *files.spec);
  }
  if (!files.shards.empty()) {
    fs::create_directories(dir / "data", error);
  }
  for (std::size_t s = 0; s < files.shards.size(); ++s) {
    std::string path = std::to_string(s);
    path.insert(0, 5 - path.size(), '0');
    WriteFile(dir / "data" / (path + ".safetensors"), files.shards[s]);
  }
  if (!files.meta.empty()) {
    fs::create_directories(dir / "meta", error);
  }
  for (const auto& [name, bytes] : files.meta) {
    WriteFile(dir / "meta" / name, bytes);
  }
  return dir;
}

// ---------------------------------------------------------- composition

CompositionFiles Render(const CompositionPlan& plan) {
  Json::Array components;
  for (const auto& [role, id, architecture] : plan.components) {
    components.emplace_back(
        Json::Object{{"architecture", architecture}, {"artifact", id}, {"role", role}});
  }
  auto meta = plan.meta;
  std::ranges::sort(meta);
  Json::Array sources;
  for (const auto& [name, bytes] : meta) {
    sources.emplace_back(
        Json::Object{{"bytes", bytes.size()}, {"name", name}, {"sha256", Hex(bytes)}});
  }
  CompositionFiles files;
  files.manifest = Json(Json::Object{
      {"components", Json(std::move(components))},
      {"experimental", true},
      {"files", Json(Json::Array{})},
      {"format", "llmp-composition"},
      {"format_version", 1},
      {"importer", Importer("llmp-import-test", "1", "test")},
      {"model", Json(Json::Object{{"architecture", plan.architecture}})},
      {"source", Json(std::move(sources))},
  });
  files.spec = plan.spec;
  files.meta = std::move(meta);
  return files;
}

fs::path Publish(const fs::path& store, const CompositionFiles& files, const PublishHooks& hooks) {
  Json manifest = files.manifest;
  if (!files.keep_files) {
    Json::Array entries;
    for (const auto& [name, bytes] : files.meta) {
      entries.push_back(FileEntry("meta/" + name, "source-metadata", bytes));
    }
    if (files.spec) {
      entries.push_back(FileEntry("spec.json", "spec", *files.spec));
    }
    manifest["files"] = Json(std::move(entries));
  }
  std::string text = manifest.Text();
  if (hooks.manifest) {
    hooks.manifest(text);
  }
  const fs::path dir = store / Hex(text);
  std::error_code error;
  fs::remove_all(dir, error);
  fs::create_directories(dir, error);
  WriteFile(dir / "manifest.json", text);
  if (files.spec) {
    WriteFile(dir / "spec.json", *files.spec);
  }
  if (!files.meta.empty()) {
    fs::create_directories(dir / "meta", error);
  }
  for (const auto& [name, bytes] : files.meta) {
    WriteFile(dir / "meta" / name, bytes);
  }
  return dir;
}

}  // namespace llmp::test
