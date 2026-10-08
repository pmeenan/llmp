// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The v0 artifact reader (src/artifact): its strict JSON and hex digests, read
// planning on hand-built layouts, and Artifact::Open against the corpus the
// M0 prototype builds and judges (artifact_corpus.py): the prototype's views
// of valid artifacts, its verdict on every mutated one, links and special
// files made here, caps on sparse files, the owner policy, direct-read
// opens and a mutation fuzz the prototype judges (ASan in check:full). The
// real v0 fixtures are opened where they are installed (a Spark) and
// skipped elsewhere.

#include "artifact/artifact.h"

#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <sstream>
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
#include "base/sha256.h"

namespace {

namespace fs = std::filesystem;
using llmp::artifact::Artifact;
using llmp::artifact::ChunkKey;
using llmp::artifact::Error;
using llmp::artifact::Rule;
using llmp::artifact::RuleName;
using llmp::base::Bytes;
namespace artifact = llmp::artifact;
namespace json = llmp::artifact::json;

// ---------------------------------------------------------------- helpers

std::string Sha256Hex(std::string_view bytes) {
  return llmp::base::ToHex(llmp::base::Sha256().Update(bytes).Finish());
}

std::string Env(const char* name) {
  const char* value = std::getenv(name);  // NOLINT(concurrency-mt-unsafe)
  return value == nullptr ? std::string() : std::string(value);
}

// The corpus is built beside the test, where CTest runs it; a manual run
// elsewhere names it in LLMP_TEST_ARTIFACT_CORPUS.
fs::path Corpus() {
  const std::string configured = Env("LLMP_TEST_ARTIFACT_CORPUS");
  return configured.empty() ? fs::path("artifact-corpus") : fs::path(configured);
}

fs::path Scratch(std::string_view name) {
  const std::string base = Env("LLMP_TEST_SCRATCH");
  const fs::path root = base.empty() ? fs::path(::testing::TempDir()) : fs::path(base);
  const fs::path dir = root / "artifact" / name;
  std::error_code error;
  fs::remove_all(dir, error);
  fs::create_directories(dir, error);
  return dir;
}

std::string ReadFile(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void WriteFile(const fs::path& path, std::string_view data) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(data.data(), static_cast<std::streamsize>(data.size()));
}

// A copy of an artifact directory under `into`, keeping its name.
fs::path CopyArtifact(const fs::path& from, const fs::path& into) {
  const fs::path to = into / from.filename();
  std::error_code error;
  fs::copy(from, to, fs::copy_options::recursive, error);
  EXPECT_FALSE(error) << error.message();
  return to;
}

std::string Verdict(const std::expected<Artifact, Error>& opened) {
  return opened ? std::string("ok") : std::string(RuleName(opened.error().rule));
}

std::string Describe(const std::expected<Artifact, Error>& opened) {
  return opened ? std::string("ok") : opened.error().ToString();
}

// The single artifact directory under dir.
fs::path OnlyChild(const fs::path& dir) {
  std::error_code error;
  for (const auto& entry : fs::directory_iterator(dir, error)) {
    return entry.path();
  }
  return {};
}

// A small mutable JSON tree for editing documents in tests, written
// canonically (sorted keys, no whitespace, one newline).
struct J {
  json::Kind kind = json::Kind::kNull;
  std::string text;               // integers (as written) and strings
  std::vector<J> items;           // arrays, or object member values
  std::vector<std::string> keys;  // object member keys

  static J From(json::Value v) {
    J out;
    out.kind = v.kind();
    switch (v.kind()) {
      case json::Kind::kInteger:
        out.text = std::string(v.raw_integer());
        break;
      case json::Kind::kString:
        out.text = std::string(v.string());
        break;
      case json::Kind::kArray:
        for (std::size_t i = 0; i < v.size(); ++i) {
          out.items.push_back(From(v.at(i)));
        }
        break;
      case json::Kind::kObject:
        for (std::size_t i = 0; i < v.size(); ++i) {
          out.keys.emplace_back(v.key(i));
          out.items.push_back(From(v.member(i)));
        }
        break;
      default:
        break;
    }
    return out;
  }
  static J Of(json::Kind kind) {
    J out;
    out.kind = kind;
    return out;
  }
  static J Int(std::int64_t n) {
    J out = Of(json::Kind::kInteger);
    out.text = std::to_string(n);
    return out;
  }
  static J Str(std::string s) {
    J out = Of(json::Kind::kString);
    out.text = std::move(s);
    return out;
  }

  J& operator[](std::string_view key) {
    for (std::size_t i = 0; i < keys.size(); ++i) {
      if (keys[i] == key) {
        return items[i];
      }
    }
    keys.emplace_back(key);
    items.emplace_back();
    return items.back();
  }

  void Dump(std::string& out) const {
    switch (kind) {
      case json::Kind::kNull:
        out += "null";
        return;
      case json::Kind::kFalse:
        out += "false";
        return;
      case json::Kind::kTrue:
        out += "true";
        return;
      case json::Kind::kInteger:
        out += text;
        return;
      case json::Kind::kString:
        out += '"' + text + '"';
        return;
      case json::Kind::kArray:
        out += '[';
        for (std::size_t i = 0; i < items.size(); ++i) {
          out += i > 0 ? "," : "";
          items[i].Dump(out);
        }
        out += ']';
        return;
      case json::Kind::kObject: {
        std::vector<std::size_t> order(keys.size());
        for (std::size_t i = 0; i < order.size(); ++i) {
          order[i] = i;
        }
        std::ranges::sort(order, [&](std::size_t a, std::size_t b) { return keys[a] < keys[b]; });
        out += '{';
        for (std::size_t i = 0; i < order.size(); ++i) {
          out += i > 0 ? ",\"" : "\"";
          out += keys[order[i]] + "\":";
          items[order[i]].Dump(out);
        }
        out += '}';
        return;
      }
    }
  }
  std::string Canonical() const {
    std::string out;
    Dump(out);
    return out + "\n";
  }
};

J Load(const fs::path& path, std::string& keep) {
  keep = ReadFile(path);
  auto doc = json::Parse(keep);
  EXPECT_TRUE(doc.has_value());
  return doc ? J::From(doc->root()) : J{};
}

// Rewrites an artifact's manifest: refreshes every listed file's size and
// digest (as the prototype's relist does), applies `edit`, and renames the
// directory to the new digest. Returns the new root.
template <typename Edit>
fs::path Relist(const fs::path& root, Edit edit) {
  std::string keep;
  J manifest = Load(root / "manifest.json", keep);
  for (J& f : manifest["files"].items) {
    const fs::path path = root / f["path"].text;
    std::error_code error;
    if (fs::exists(path, error) && fs::file_size(path, error) < (std::uint64_t{1} << 24U)) {
      const std::string data = ReadFile(path);
      f["bytes"] = J::Int(static_cast<std::int64_t>(data.size()));
      f["sha256"] = J::Str(Sha256Hex(data));
    }
  }
  edit(manifest);
  const std::string bytes = manifest.Canonical();
  WriteFile(root / "manifest.json", bytes);
  const fs::path renamed = root.parent_path() / Sha256Hex(bytes);
  std::error_code error;
  fs::rename(root, renamed, error);
  EXPECT_FALSE(error) << error.message();
  return renamed;
}

// Adds a listed file entry to a manifest, keeping files sorted by path.
void AddFile(J& manifest, const std::string& path, std::string_view role, std::uint64_t bytes) {
  J entry = J::Of(json::Kind::kObject);
  entry["path"] = J::Str(path);
  entry["role"] = J::Str(std::string(role));
  entry["bytes"] = J::Int(static_cast<std::int64_t>(bytes));
  entry["sha256"] = J::Str(std::string(64, '0'));
  auto& files = manifest["files"].items;
  files.push_back(entry);
  std::ranges::sort(files, [](J& a, J& b) { return a["path"].text < b["path"].text; });
}

// Records a source (the kept files' coverage), keeping sources sorted by name.
void AddSource(J& manifest, const std::string& name, std::uint64_t bytes) {
  J entry = J::Of(json::Kind::kObject);
  entry["name"] = J::Str(name);
  entry["bytes"] = J::Int(static_cast<std::int64_t>(bytes));
  entry["sha256"] = J::Str(std::string(64, '0'));
  auto& sources = manifest["source"].items;
  sources.push_back(entry);
  std::ranges::sort(sources, [](J& a, J& b) { return a["name"].text < b["name"].text; });
}

void MakeSparse(const fs::path& path, std::uint64_t size) {
  WriteFile(path, "");
  std::error_code error;
  fs::resize_file(path, size, error);
  EXPECT_FALSE(error) << error.message();
}

// ---------------------------------------------------------------- the view

std::string Join(const std::vector<std::uint64_t>& values, std::string_view separator) {
  std::string out;
  for (std::size_t i = 0; i < values.size(); ++i) {
    out += (i > 0 ? std::string(separator) : std::string()) + std::to_string(values[i]);
  }
  return out;
}

std::string ReprText(const artifact::Representation& r) {
  if (r.family == artifact::Family::kGgml) {
    return "ggml:" + std::string(r.type) + ":" + Join(r.dims, "x");
  }
  std::string text = std::string(artifact::FamilyName(r.family)) + ":" +
                     (r.role.empty() ? std::string("-") : std::string(r.role)) + ":" +
                     std::string(r.type) + ":" + Join(r.dims, "x");
  if (r.role == "trellis") {
    text += ":k" + std::to_string(r.k_bits) + ":in" + std::to_string(r.in_features) + ":out" +
            std::to_string(r.out_features) + ":" + std::string(r.codebook);
  }
  return text;
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
  return "?";
}

std::string_view RoleName(artifact::FileRole role) {
  switch (role) {
    case artifact::FileRole::kIndex:
      return "index";
    case artifact::FileRole::kShard:
      return "shard";
    case artifact::FileRole::kSourceMetadata:
      return "source-metadata";
  }
  return "?";
}

std::string Dash(const std::optional<std::uint32_t>& v) {
  return v ? std::to_string(*v) : std::string("-");
}

// The same facts, in the same form, as artifact_corpus.py's view().
std::string View(const Artifact& a) {
  std::ostringstream out;
  out << "artifact " << a.id() << "\n";
  std::string families;
  for (const artifact::Family f : a.model().representation) {
    families += (families.empty() ? "" : ",") + std::string(artifact::FamilyName(f));
  }
  out << "model " << a.model().architecture << " " << a.model().expert_count << " " << families
      << "\n";
  for (const auto& s : a.sources()) {
    out << "source " << s.name << " " << s.bytes.value() << " " << llmp::base::ToHex(s.sha256)
        << "\n";
  }
  for (const auto& f : a.files()) {
    out << "file " << f.path << " " << RoleName(f.role) << " " << f.bytes.value() << " "
        << llmp::base::ToHex(f.sha256) << "\n";
  }
  for (std::size_t i = 0; i < a.shards().size(); ++i) {
    const auto& s = a.shards()[i];
    out << "shard " << i << " " << s.path << " " << s.data_offset.value() << " "
        << s.data_bytes.value() << " " << llmp::base::ToHex(s.header_sha256) << "\n";
  }
  std::vector<ChunkKey> everything;
  for (std::uint32_t gid = 0; gid < a.groups().size(); ++gid) {
    const auto& g = a.groups()[gid];
    out << "group " << gid << " " << KindName(g.kind) << " " << Dash(g.layer) << " "
        << Dash(g.expert) << " " << g.shard << " " << g.offset.value() << " " << g.used.value()
        << " " << g.stored.value() << " " << g.first_chunk << " " << g.chunks << "\n";
    for (std::uint32_t k = 0; k < g.chunks; ++k) {
      const auto range = artifact::ChunkRangeOf(a.layout(), {.group = gid, .chunk = k});
      EXPECT_TRUE(range.has_value());
      out << "chunk " << gid << " " << k << " " << a.shards()[range->shard].path << " "
          << range->file_offset.value() << " " << range->length.value() << " "
          << llmp::base::ToHex(a.chunk_sha256()[g.first_chunk + k]) << "\n";
      everything.push_back({.group = gid, .chunk = k});
    }
  }
  for (std::uint32_t i = 0; i < a.resources().size(); ++i) {
    const auto& r = a.resources()[i];
    std::string roles;
    for (const auto& role : r.roles) {
      roles += (roles.empty() ? "" : ",") + role;
    }
    const auto placement = a.ResourcePlacement(i);
    EXPECT_TRUE(placement.has_value());
    out << "resource " << r.name << " " << roles << " " << r.group << " " << r.offset.value() << " "
        << r.bytes.value() << " " << r.readable.value() << " " << (r.rows ? "rows" : "-") << " "
        << ReprText(r.repr) << " " << placement->closure.first << " " << placement->closure.last
        << "\n";
    if (r.rows) {
      const auto geometry = a.Rows(i);
      EXPECT_TRUE(geometry.has_value());
      const std::set<std::uint64_t> unique = {0, geometry->rows / 2, geometry->rows - 1};
      const std::vector<std::uint64_t> rows(unique.begin(), unique.end());
      const auto keys = a.RowChunks(i, rows);
      EXPECT_TRUE(keys.has_value());
      std::string chunks;
      for (const auto& key : *keys) {
        chunks += (chunks.empty() ? "" : ",") + std::to_string(key.group) + ":" +
                  std::to_string(key.chunk);
      }
      out << "rows " << r.name << " " << geometry->rows << " " << geometry->row_bytes << " "
          << Join(rows, ",") << " " << chunks << "\n";
    }
  }
  for (std::uint32_t i = 0; i < a.expert_arrays().size(); ++i) {
    const auto& x = a.expert_arrays()[i];
    out << "array " << x.name << " " << x.layer << " " << x.count << " " << x.first_group << " "
        << x.group_offset.value() << " " << x.slice_bytes.value() << " " << x.readable.value()
        << " " << ReprText(x.repr) << "\n";
    for (std::uint32_t e = 0; e < x.count; ++e) {
      const auto placement = a.SlicePlacement(i, e);
      EXPECT_TRUE(placement.has_value());
      out << "slice " << x.name << " " << e << " " << placement->group << " "
          << placement->closure.first << " " << placement->closure.last << "\n";
    }
  }
  std::vector<ChunkKey> odd;
  std::vector<ChunkKey> even;
  for (std::size_t i = 0; i < everything.size(); ++i) {
    (i % 2 == 0 ? even : odd).push_back(everything[i]);
  }
  struct Plan {
    std::string_view label;
    const std::vector<ChunkKey>* missing;
    const std::vector<ChunkKey>* resident;
    std::uint64_t max_run;
    std::size_t max_segments;
  };
  const std::vector<ChunkKey> none;
  for (const Plan& p : {Plan{"all", &everything, &none, std::uint64_t{64} << 20U, 1024},
                        Plan{"all", &everything, &none, std::uint64_t{4} << 20U, 3},
                        Plan{"all", &everything, &none, 1, 1},
                        Plan{"alternate", &even, &odd, std::uint64_t{64} << 20U, 1024}}) {
    const auto runs = a.PlanReads(*p.missing, *p.resident,
                                  {.max_run = Bytes(p.max_run), .max_segments = p.max_segments});
    EXPECT_TRUE(runs.has_value());
    for (const auto& run : *runs) {
      out << "run " << p.label << " " << p.max_run << " " << p.max_segments << " "
          << a.shards()[run.shard].path << " " << run.file_offset.value() << " "
          << run.length.value();
      for (const auto& s : run.segments) {
        out << " " << s.chunk.group << ":" << s.chunk.chunk << ":" << s.length.value();
        EXPECT_EQ(s.group_offset.value(), std::uint64_t{s.chunk.chunk} * artifact::kChunkBytes);
      }
      out << "\n";
    }
  }
  return out.str();
}

// ---------------------------------------------------------------- digests

TEST(ArtifactSchemaTest, DigestsFromHex) {
  const std::string hex = Sha256Hex("abc");
  EXPECT_EQ(
      llmp::base::ToHex(artifact::schema::DigestFromHex(hex).value_or(llmp::base::Sha256Digest{})),
      hex);
  EXPECT_FALSE(artifact::schema::DigestFromHex(std::string(63, 'a')).has_value());
  EXPECT_FALSE(artifact::schema::DigestFromHex(std::string(65, 'a')).has_value());
  EXPECT_FALSE(artifact::schema::DigestFromHex(std::string(64, 'A')).has_value());
  EXPECT_FALSE(artifact::schema::DigestFromHex(std::string(64, 'g')).has_value());
}

// ---------------------------------------------------------------- JSON

struct Parsed {
  std::string verdict;  // "canonical", "ok" (non-canonical) or the error's rule
};

Parsed ParseVerdict(std::string_view text) {
  const auto doc = json::Parse(text);
  if (!doc) {
    return {std::string(RuleName(doc.error().rule))};
  }
  return {doc->canonical() ? "canonical" : "ok"};
}

TEST(ArtifactJsonTest, AcceptsExactlyTheCanonicalSubset) {
  const std::vector<std::pair<std::string_view, std::string_view>> cases = {
      {"{}\n", "canonical"},
      {"[]\n", "canonical"},
      {R"({"a":1,"b":[true,false,null,"x",-5]})"
       "\n",
       "canonical"},
      {"{}", "ok"},                   // no trailing newline
      {"{}\n\n", "ok"},               // two
      {" {}\n", "ok"},                // leading whitespace
      {"{\"a\": 1}\n", "ok"},         // whitespace between tokens
      {"{\"b\":1,\"a\":2}\n", "ok"},  // unsorted keys
      {"[-0]\n", "ok"},               // "-0" is written "0"
      {"[\"a\x7f\"]\n", "ok"},        // the canonical encoder escapes DEL
      {"{\"a\":1,\"a\":2}\n", "json"},
      {"{\"b\":1,\"a\":2,\"b\":3}\n", "json"},
      {"[1.0]\n", "json"},
      {"[1e5]\n", "json"},
      {"[1E5]\n", "json"},
      {"[01]\n", "json"},
      {"[-]\n", "json"},
      {"[NaN]\n", "json"},
      {"[Infinity]\n", "json"},
      {"[-Infinity]\n", "json"},
      {"[tru]\n", "json"},
      {"[1,]\n", "json"},
      {"{\"a\":1,}\n", "json"},
      {"{\"a\"}\n", "json"},
      {"{1:2}\n", "json"},
      {"[1]x\n", "json"},
      {"[1] [2]\n", "json"},
      {"", "json"},
      {"\n", "json"},
      {"[\"a\tb\"]\n", "json"},
      {"[\"a\\nb\"]\n", "canonical-error"},
      {"[\"\xc3\xa9\"]\n", "json"},
      {"\xef\xbb\xbf{}\n", "json"},
      {"[\"unterminated]\n", "json"},
      {"[12345678901234567890]\n", "canonical"},  // 20 characters: parses, out of range
      {"[-1234567890123456789]\n", "canonical"},  // 20 characters
      {"[123456789012345678901]\n", "json"},      // 21
      {"[[[[[[[[1]]]]]]]]\n", "canonical"},       // depth 8
      {"[[[[[[[[[1]]]]]]]]]\n", "json"},          // depth 9
  };
  for (const auto& [text, want] : cases) {
    const std::string got = ParseVerdict(text).verdict;
    EXPECT_EQ(got, want == "canonical-error" ? "canonical" : want) << text;
  }
}

TEST(ArtifactJsonTest, TypedAccess) {
  const std::string text =
      R"({"a":9223372036854775807,"b":-9223372036854775808,"c":9223372036854775808,"d":"s","e":[1,2]})"
      "\n";
  const auto doc = json::Parse(text);
  ASSERT_TRUE(doc.has_value());
  const json::Value root = doc->root();
  const auto member = [&](std::string_view key) {
    const auto found = root.find(key);
    EXPECT_TRUE(found.has_value()) << key;
    return found.value_or(root);
  };
  EXPECT_EQ(member("a").integer(), INT64_MAX);
  EXPECT_EQ(member("b").integer(), INT64_MIN);
  EXPECT_FALSE(member("c").integer().has_value());  // beyond 64 bits: never a valid field
  EXPECT_EQ(member("c").raw_integer(), "9223372036854775808");
  EXPECT_EQ(member("d").string(), "s");
  EXPECT_FALSE(member("d").integer().has_value());
  EXPECT_EQ(member("e").size(), 2U);
  EXPECT_FALSE(root.find("z").has_value());
  std::string out;
  json::Serialize(root, out);
  EXPECT_EQ(out + "\n", text);
}

TEST(ArtifactJsonTest, CapsAreEnforced) {
  // 2^19 containers are allowed, one more is not.
  std::string arrays = "[";
  for (std::size_t i = 0; i + 1 < json::kMaxContainers; ++i) {
    arrays += i > 0 ? ",[]" : "[]";
  }
  EXPECT_EQ(ParseVerdict(arrays + "]\n").verdict, "canonical");
  EXPECT_EQ(ParseVerdict(arrays + ",[]]\n").verdict, "json");
  // 2^21 separators are allowed, one more is not.
  std::string zeros = "[0";
  for (std::size_t i = 0; i < json::kMaxSeparators; ++i) {
    zeros += ",0";
  }
  EXPECT_EQ(ParseVerdict(zeros + "]\n").verdict, "canonical");
  EXPECT_EQ(ParseVerdict(zeros + ",0]\n").verdict, "json");
}

TEST(ArtifactJsonTest, CanonicalDocumentsRoundTrip) {
  // Any document the parser calls canonical re-serializes to its own bytes.
  const std::string index = ReadFile(OnlyChild(Corpus() / "golden" / "moe") / "index.json");
  ASSERT_FALSE(index.empty());
  std::mt19937 random(11);  // NOLINT(bugprone-random-generator-seed): reproducible
  for (int i = 0; i < 2000; ++i) {
    std::string text = index;
    const int edits = 1 + static_cast<int>(random() % 4);
    for (int e = 0; e < edits && !text.empty(); ++e) {
      const std::size_t at = random() % text.size();
      switch (random() % 5) {
        case 0:
          text[at] = static_cast<char>(random() & 0xFFU);
          break;
        case 1:
          text.erase(at, 1 + (random() % 16));
          break;
        case 2:
          text.insert(at, 1, "{}[],:\"0-1 \\\x7f"[random() % 13]);
          break;
        case 3:
          text.resize(at);
          break;
        default:
          text.insert(at, text.substr(random() % text.size(), random() % 64));
          break;
      }
    }
    const auto doc = json::Parse(text);
    if (doc && doc->canonical()) {
      std::string out;
      json::Serialize(doc->root(), out);
      EXPECT_EQ(out + "\n", text);
    }
  }
}

// ---------------------------------------------------------------- planning

artifact::Layout Groups(const std::vector<std::uint64_t>& sizes) {
  artifact::Layout layout;
  std::uint64_t offset = 0;
  std::uint64_t chunk = 0;
  for (const std::uint64_t n : sizes) {
    const auto chunks =
        static_cast<std::uint32_t>((n + artifact::kChunkBytes - 1) / artifact::kChunkBytes);
    layout.groups.push_back({.kind = artifact::GroupKind::kLayer,
                             .layer = 0,
                             .expert = std::nullopt,
                             .shard = 0,
                             .offset = Bytes(offset),
                             .used = Bytes(n),
                             .stored = Bytes(n),
                             .first_chunk = chunk,
                             .chunks = chunks});
    offset += n;
    chunk += chunks;
  }
  layout.shards.push_back({.path = "data/00000.safetensors",
                           .data_offset = Bytes(4096),
                           .data_bytes = Bytes(offset),
                           .header_sha256 = {}});
  return layout;
}

constexpr std::uint64_t kChunk = artifact::kChunkBytes;

TEST(ArtifactLayoutTest, ClosureCoversTheRangeWithinItsGroup) {
  const auto layout = Groups({3 * kChunk});
  const auto c = artifact::ClosureOf(layout, 0, Bytes(kChunk - 256), Bytes(512));
  ASSERT_TRUE(c.has_value());
  EXPECT_EQ(std::pair(c->first, c->last), std::pair(0U, 1U));
  const auto whole = artifact::ClosureOf(layout, 0, Bytes(0), Bytes(kChunk));
  ASSERT_TRUE(whole.has_value());
  EXPECT_EQ(std::pair(whole->first, whole->last), std::pair(0U, 0U));
  EXPECT_FALSE(artifact::ClosureOf(layout, 0, Bytes(0), Bytes(4 * kChunk)).has_value());
  EXPECT_FALSE(artifact::ClosureOf(layout, 1, Bytes(0), Bytes(1)).has_value());
  EXPECT_FALSE(artifact::ClosureOf(layout, 0, Bytes(0), Bytes(0)).has_value());
  EXPECT_FALSE(artifact::ClosureOf(layout, 0, Bytes(UINT64_MAX), Bytes(2)).has_value());
}

TEST(ArtifactLayoutTest, CoalescesFileAdjacentChunksAcrossGroups) {
  const auto layout = Groups({kChunk + 8192, 4096, 2 * kChunk});
  const std::vector<ChunkKey> missing = {{2, 0}, {0, 1}, {1, 0}, {0, 0}, {0, 0}};
  const auto runs = artifact::PlanReads(layout, missing, {});
  ASSERT_TRUE(runs.has_value());
  ASSERT_EQ(runs->size(), 1U);
  EXPECT_EQ((*runs)[0].file_offset.value(), 4096U);
  EXPECT_EQ((*runs)[0].length.value(), kChunk + 8192 + 4096 + kChunk);
  std::vector<std::uint64_t> lengths;
  for (const auto& s : (*runs)[0].segments) {
    lengths.push_back(s.length.value());
  }
  EXPECT_EQ(lengths, (std::vector<std::uint64_t>{kChunk, 8192, 4096, kChunk}));
  EXPECT_EQ((*runs)[0].segments[1].group_offset.value(), kChunk);  // chunk 1 of group 0
}

TEST(ArtifactLayoutTest, NeverBridgesResidentChunks) {
  const auto layout = Groups({3 * kChunk});
  const std::vector<ChunkKey> missing = {{0, 0}, {0, 2}};
  const std::vector<ChunkKey> resident = {{0, 1}};
  const auto runs = artifact::PlanReads(layout, missing, resident);
  ASSERT_TRUE(runs.has_value());
  EXPECT_EQ(runs->size(), 2U);
  const std::vector<ChunkKey> both = {{0, 1}};
  const auto conflict = artifact::PlanReads(layout, both, resident);
  ASSERT_FALSE(conflict.has_value());
  EXPECT_EQ(conflict.error().rule, Rule::kState);
}

TEST(ArtifactLayoutTest, RejectsOutOfRangeChunks) {
  const auto layout = Groups({kChunk, kChunk});
  for (const ChunkKey key : {ChunkKey{0, 1}, ChunkKey{2, 0}, ChunkKey{1, 5},
                             ChunkKey{UINT32_MAX, 0}, ChunkKey{0, UINT32_MAX}}) {
    const std::vector<ChunkKey> missing = {key};
    const auto runs = artifact::PlanReads(layout, missing, {});
    ASSERT_FALSE(runs.has_value());
    EXPECT_EQ(runs.error().rule, Rule::kBounds);
  }
}

TEST(ArtifactLayoutTest, BoundsRunSizeAndSegments) {
  const auto layout = Groups({8 * kChunk});
  std::vector<ChunkKey> all;
  all.reserve(8);
  for (std::uint32_t k = 0; k < 8; ++k) {
    all.push_back({0, k});
  }
  const auto runs = artifact::PlanReads(layout, all, {}, {.max_run = Bytes(4 * kChunk)});
  ASSERT_TRUE(runs.has_value());
  ASSERT_EQ(runs->size(), 2U);
  EXPECT_EQ((*runs)[0].length.value(), 4 * kChunk);
  std::vector<std::uint64_t> small(1025, 4096);
  const auto many = Groups(small);
  std::vector<ChunkKey> keys;
  keys.reserve(1025);
  for (std::uint32_t g = 0; g < 1025; ++g) {
    keys.push_back({g, 0});
  }
  const auto split = artifact::PlanReads(many, keys, {});
  ASSERT_TRUE(split.has_value());
  ASSERT_EQ(split->size(), 2U);
  EXPECT_EQ((*split)[0].segments.size(), 1024U);
  EXPECT_EQ((*split)[1].segments.size(), 1U);
}

// ---------------------------------------------------------------- the corpus

TEST(ArtifactCorpusTest, TypeTablesMatchThePrototype) {
  std::istringstream lines(ReadFile(Corpus() / "types.txt"));
  ASSERT_FALSE(lines.str().empty()) << "no corpus at " << Corpus();
  std::size_t ggml = 0;
  std::size_t dtypes = 0;
  std::size_t native = 0;
  std::string kind;
  while (lines >> kind) {
    std::string name;
    lines >> name;
    if (kind == "ggml") {
      std::uint32_t id = 0;
      std::uint32_t block = 0;
      std::uint32_t bytes = 0;
      lines >> id >> block >> bytes;
      const auto* type = artifact::FindGgmlType(name);
      ASSERT_NE(type, nullptr) << name;
      EXPECT_EQ(std::tuple(type->id, type->block_elements, type->block_bytes),
                std::tuple(id, block, bytes))
          << name;
      ++ggml;
    } else if (kind == "dtype") {
      std::uint32_t bytes = 0;
      lines >> bytes;
      const auto* dtype = artifact::FindDtype(name);
      ASSERT_NE(dtype, nullptr) << name;
      EXPECT_EQ(dtype->bytes, bytes) << name;
      ++dtypes;
    } else {
      EXPECT_TRUE(artifact::IsNativeContainerType(name)) << name;
      ++native;
    }
  }
  EXPECT_EQ(ggml, artifact::GgmlTypes().size());
  EXPECT_EQ(dtypes, artifact::Dtypes().size());
  EXPECT_EQ(native, 8U);
}

TEST(ArtifactCorpusTest, ViewsMatchThePrototype) {
  for (const char* name : {"tiny", "exl3", "moe"}) {
    const fs::path root = OnlyChild(Corpus() / "golden" / name);
    ASSERT_FALSE(root.empty()) << "no corpus at " << Corpus();
    const auto opened = Artifact::Open(root);
    ASSERT_TRUE(opened.has_value()) << name << ": " << Describe(opened);
    const std::string want = ReadFile(Corpus() / "golden" / (std::string(name) + ".view"));
    EXPECT_EQ(View(*opened), want) << name;
  }
}

// Where the reader's rule knowingly differs from the prototype's on the same
// input (both refuse, or the reader accepts what it does not read):
//   - a string escape is refused outright, whatever the field; the format's
//     strings never contain `"` or `\`, while the prototype refuses the
//     document under the first rule it reaches (`json`, `canonical`, or the
//     field's rule);
//   - the reader does not parse kept `.kv.gguf` metadata (artifact.h), which
//     the prototype's loader-open does: it accepts what only that refuses,
//     and refuses a manifest expert count that also disagrees with the
//     expert arrays under the prototype's next rule.
const std::map<std::string, std::string, std::less<>> kDivergences = {
    {"manifest-escaped-newline", "canonical"},
    {"meta-empty-kv", "ok"},
    {"manifest-expert-count", "expert-array"},
};

TEST(ArtifactCorpusTest, VerdictsMatchThePrototype) {
  std::istringstream lines(ReadFile(Corpus() / "cases.txt"));
  std::string name;
  std::string id;
  std::string oracle;
  std::size_t cases = 0;
  std::set<std::string> rules;
  while (lines >> name >> id >> oracle) {
    const auto opened = Artifact::Open(Corpus() / "cases" / name / id);
    const auto divergence = kDivergences.find(name);
    const std::string want = divergence != kDivergences.end() ? divergence->second : oracle;
    EXPECT_EQ(Verdict(opened), want)
        << name << ": " << Describe(opened) << " (prototype: " << oracle << ")";
    rules.insert(oracle);
    ++cases;
  }
  EXPECT_GE(cases, 150U) << "no corpus at " << Corpus();
  // Every rule the loader-open path can name has cases.
  for (const char* rule :
       {"ok", "unsupported-version", "unsupported-profile", "format", "json", "canonical",
        "identity", "schema", "path", "file-set", "file-size", "hash", "bounds", "alignment",
        "overlap", "repr", "expert-array", "container"}) {
    EXPECT_TRUE(rules.contains(rule)) << rule;
  }
}

// ---------------------------------------------------------------- the directory

class ArtifactFilesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    scratch_ = Scratch(::testing::UnitTest::GetInstance()->current_test_info()->name());
    root_ = CopyArtifact(OnlyChild(Corpus() / "golden" / "tiny"), scratch_);
    ASSERT_TRUE(Artifact::Open(root_).has_value());
  }
  void TearDown() override {
    std::error_code error;
    fs::permissions(root_, fs::perms::owner_all, fs::perm_options::add, error);
    fs::remove_all(scratch_, error);
  }

  static std::string Open(const fs::path& root, const artifact::OpenOptions& options = {}) {
    return Verdict(Artifact::Open(root, options));
  }

  fs::path scratch_;
  fs::path root_;
};

TEST_F(ArtifactFilesTest, LinksAndSpecialFilesAreRefused) {
  std::error_code error;
  // The root through a link.
  fs::create_directory_symlink(root_, scratch_ / "link", error);
  EXPECT_EQ(Open(scratch_ / "link",
                 {.expected_id = root_.filename().string(), .trusted_owner = std::nullopt}),
            "file-type");
  // A link in place of a listed file, a directory or the manifest.
  const fs::path index = root_ / "index.json";
  fs::rename(index, scratch_ / "index.json", error);
  fs::create_symlink(scratch_ / "index.json", index, error);
  EXPECT_EQ(Open(root_), "file-type");
  fs::remove(index, error);
  fs::rename(scratch_ / "index.json", index, error);
  EXPECT_EQ(Open(root_), "ok");
  fs::rename(root_ / "meta", scratch_ / "meta", error);
  fs::create_directory_symlink(scratch_ / "meta", root_ / "meta", error);
  EXPECT_EQ(Open(root_), "file-type");
  fs::remove(root_ / "meta", error);
  fs::rename(scratch_ / "meta", root_ / "meta", error);
  const fs::path manifest = root_ / "manifest.json";
  fs::rename(manifest, scratch_ / "manifest.json", error);
  fs::create_symlink(scratch_ / "manifest.json", manifest, error);
  EXPECT_EQ(Open(root_), "file-type");
  fs::remove(manifest, error);
  EXPECT_EQ(Open(root_), "file-type");  // missing
  fs::rename(scratch_ / "manifest.json", manifest, error);
  EXPECT_EQ(Open(root_), "ok");
  // A hard-linked file.
  const fs::path shard = root_ / "data" / "00000.safetensors";
  fs::create_hard_link(shard, scratch_ / "second-name", error);
  EXPECT_EQ(Open(root_), "file-type");
  fs::remove(scratch_ / "second-name", error);
  EXPECT_EQ(Open(root_), "ok");
  // A FIFO swapped in for a listed file never blocks the open.
  const fs::path kv = root_ / "meta" / "tiny.kv.gguf";
  fs::rename(kv, scratch_ / "kv", error);
  ASSERT_EQ(::mkfifo(kv.c_str(), 0600), 0) << std::strerror(errno);  // NOLINT
  EXPECT_EQ(Open(root_), "file-type");
  fs::remove(kv, error);
  fs::rename(scratch_ / "kv", kv, error);
  // Not a directory, or nothing at all.
  EXPECT_EQ(Open(shard, {.expected_id = root_.filename().string(), .trusted_owner = std::nullopt}),
            "file-type");
  EXPECT_EQ(Open(scratch_ / "absent"), "file-type");
  EXPECT_EQ(Open(root_), "ok");
  // A trailing slash names the same directory.
  EXPECT_EQ(Open(root_.string() + "/"), "ok");
}

TEST_F(ArtifactFilesTest, UnreadableDirectoryFailsClosed) {
  if (::geteuid() == 0) {
    GTEST_SKIP() << "root reads any directory";
  }
  std::error_code error;
  fs::permissions(root_ / "meta", fs::perms::none, error);
  EXPECT_EQ(Open(root_), "file-type");
  fs::permissions(root_ / "meta", fs::perms::owner_all, error);
  EXPECT_EQ(Open(root_), "ok");
}

TEST_F(ArtifactFilesTest, SizeCapsNeedNoReads) {
  // Kept metadata over its per-file cap: a sparse file, listed with its
  // size; refused before anything reads it.
  fs::path root = root_;
  MakeSparse(root / "meta" / "big.bin", artifact::kMaxMetadataBytes + 1);
  root = Relist(root, [](J& m) {
    AddFile(m, "meta/big.bin", "source-metadata", artifact::kMaxMetadataBytes + 1);
  });
  EXPECT_EQ(Open(root), "file-size");
  // Over the total cap, each within its own and each a recorded source.
  std::error_code error;
  fs::remove(root / "meta" / "big.bin", error);
  const std::uint64_t each = std::uint64_t{50} << 20U;
  for (const char* name : {"a.bin", "b.bin", "c.bin"}) {
    MakeSparse(root / "meta" / name, each);
  }
  root = Relist(root, [&](J& m) {
    auto& files = m["files"].items;
    std::erase_if(files, [](J& f) { return f["path"].text == "meta/big.bin"; });
    for (const char* name : {"a.bin", "b.bin", "c.bin"}) {
      AddFile(m, std::string("meta/") + name, "source-metadata", each);
      AddSource(m, name, each);
    }
  });
  EXPECT_EQ(Open(root), "file-size");
  for (const char* name : {"a.bin", "b.bin", "c.bin"}) {
    fs::remove(root / "meta" / name, error);
  }
  root = Relist(root, [](J& m) {
    std::erase_if(m["files"].items, [](J& f) { return f["path"].text.ends_with(".bin"); });
    std::erase_if(m["source"].items, [](J& f) { return f["name"].text.ends_with(".bin"); });
  });
  EXPECT_EQ(Open(root), "ok");
  // An index over its cap.
  MakeSparse(root / "index.json", artifact::kMaxIndexBytes + 1);
  root = Relist(root, [](J& m) {
    for (J& f : m["files"].items) {
      if (f["path"].text == "index.json") {
        f["bytes"] = J::Int(static_cast<std::int64_t>(artifact::kMaxIndexBytes + 1));
      }
    }
  });
  EXPECT_EQ(Open(root), "file-size");
}

TEST_F(ArtifactFilesTest, TooManyDirectoryEntries) {
  for (std::uint64_t i = 0; i <= artifact::kMaxDirectoryEntries; ++i) {
    WriteFile(root_ / "meta" / ("x" + std::to_string(i)), "");
  }
  EXPECT_EQ(Open(root_), "file-set");
}

TEST_F(ArtifactFilesTest, OwnerPolicy) {
  const uid_t me = ::geteuid();
  const std::string verdict = Open(root_, {.expected_id = std::nullopt, .trusted_owner = me});
  if (verdict == "untrusted") {
    GTEST_SKIP() << "the scratch path is not private to this user here";
  }
  EXPECT_EQ(verdict, "ok");
  const fs::path shard = root_ / "data" / "00001.safetensors";
  std::error_code error;
  fs::permissions(shard, fs::perms::others_write, fs::perm_options::add, error);
  EXPECT_EQ(Open(root_, {.expected_id = std::nullopt, .trusted_owner = me}), "untrusted");
  EXPECT_EQ(Open(root_), "ok");  // without the policy, modes are not the format's business
  fs::permissions(shard, fs::perms::others_write, fs::perm_options::remove, error);
  fs::permissions(root_ / "meta", fs::perms::others_write, fs::perm_options::add, error);
  EXPECT_EQ(Open(root_, {.expected_id = std::nullopt, .trusted_owner = me}), "untrusted");
  fs::permissions(root_ / "meta", fs::perms::others_write, fs::perm_options::remove, error);
  if (me != 0) {
    // Files owned by someone other than root or the trusted user.
    EXPECT_EQ(Open(root_, {.expected_id = std::nullopt, .trusted_owner = me + 1}), "untrusted");
  }
}

TEST_F(ArtifactFilesTest, LookupsAndRowBounds) {
  const auto opened = Artifact::Open(root_);
  ASSERT_TRUE(opened.has_value());
  const Artifact& a = *opened;
  constexpr std::uint32_t kNone = UINT32_MAX;
  const std::uint32_t table = a.FindResource("token_embd.weight").value_or(kNone);
  ASSERT_NE(table, kNone);
  EXPECT_EQ(a.FindResource("output.weight"), table);  // the tied alias
  EXPECT_FALSE(a.FindResource("nope").has_value());
  const std::uint32_t norm = a.FindResource("output_norm.weight").value_or(kNone);
  ASSERT_NE(norm, kNone);
  const auto rows = a.Rows(table).value_or(artifact::RowGeometry{});
  EXPECT_EQ(rows.rows, 8U);
  EXPECT_EQ(rows.row_bytes, 512U);
  const std::vector<std::uint64_t> outside = {0, rows.rows};
  const auto refused = a.RowChunks(table, outside);
  ASSERT_FALSE(refused.has_value());
  EXPECT_EQ(refused.error().rule, Rule::kBounds);
  const std::vector<std::uint64_t> first = {0};
  EXPECT_FALSE(a.RowChunks(norm, first).has_value());  // not a row table
  EXPECT_FALSE(a.Rows(static_cast<std::uint32_t>(a.resources().size())).has_value());
  EXPECT_FALSE(a.ResourcePlacement(static_cast<std::uint32_t>(a.resources().size())).has_value());
  const std::uint32_t array = a.FindExpertArray("blk.1.ffn_down_exps.weight").value_or(kNone);
  ASSERT_NE(array, kNone);
  EXPECT_TRUE(a.SlicePlacement(array, 1).has_value());
  EXPECT_FALSE(a.SlicePlacement(array, 2).has_value());
  EXPECT_FALSE(a.SlicePlacement(99, 0).has_value());
}

TEST_F(ArtifactFilesTest, DirectReadsOpenTheValidatedShard) {
  const auto opened = Artifact::Open(root_);
  ASSERT_TRUE(opened.has_value());
  const Artifact& a = *opened;
  auto fd = a.OpenShardForDirectRead(0);
  if (!fd && fd.error().rule == Rule::kIo) {
    GTEST_SKIP() << "no direct I/O on this filesystem: " << fd.error().ToString();
  }
  ASSERT_TRUE(fd.has_value()) << fd.error().ToString();
  EXPECT_EQ(::fcntl(fd->get(), F_GETFL) & O_NONBLOCK, 0);
  EXPECT_NE(::fcntl(fd->get(), F_GETFL) & O_DIRECT, 0);
  const auto range = artifact::ChunkRangeOf(a.layout(), {.group = 1, .chunk = 0});
  ASSERT_TRUE(range.has_value());
  void* buffer = nullptr;
  ASSERT_EQ(::posix_memalign(&buffer, 4096, range->length.value()), 0);
  const ssize_t n = ::pread(fd->get(), buffer, range->length.value(),
                            static_cast<off_t>(range->file_offset.value()));
  ASSERT_EQ(n, static_cast<ssize_t>(range->length.value()));
  const std::string file = ReadFile(root_ / a.shards()[range->shard].path);
  EXPECT_EQ(std::memcmp(buffer, file.data() + range->file_offset.value(), range->length.value()),
            0);
  std::free(buffer);  // NOLINT(cppcoreguidelines-no-malloc)
  EXPECT_FALSE(a.OpenShardForDirectRead(99).has_value());
  // A shard rewritten in place after open (here with its own bytes) is not
  // the one validated: its status-change time moved. On a kernel without
  // multigrain timestamps, or a file system with coarse ones (ext4 with
  // 128-byte inodes counts whole seconds), a rewrite within one tick may
  // not move it, so that case is only judged when it did.
  const fs::path second = root_ / "data" / "00001.safetensors";
  struct stat before{};
  ASSERT_EQ(::stat(second.c_str(), &before), 0);
  const std::string content = ReadFile(second);
  WriteFile(second, content);
  struct stat after{};
  ASSERT_EQ(::stat(second.c_str(), &after), 0);
  ASSERT_EQ(after.st_ino, before.st_ino);
  if (after.st_ctim.tv_sec != before.st_ctim.tv_sec ||
      after.st_ctim.tv_nsec != before.st_ctim.tv_nsec) {
    const auto rewritten = a.OpenShardForDirectRead(1);
    ASSERT_FALSE(rewritten.has_value());
    EXPECT_EQ(rewritten.error().rule, Rule::kFileType);
  }
  // A shard deleted and created again with the same size is not the one
  // validated, even where the file system reuses its inode number (ext4
  // does, on the Sparks) and its timestamps are too coarse to tell (the
  // inode generation does: RE-023, checked on a 128-byte-inode ext4).
  const fs::path third = root_ / "data" / "00002.safetensors";
  struct stat old_third{};
  ASSERT_EQ(::stat(third.c_str(), &old_third), 0);
  std::string different = ReadFile(third);
  different.back() = static_cast<char>(different.back() ^ 1);
  ASSERT_EQ(::unlink(third.c_str()), 0);
  WriteFile(third, different);
  struct stat new_third{};
  ASSERT_EQ(::stat(third.c_str(), &new_third), 0);
  const auto recreated = a.OpenShardForDirectRead(2);
  ASSERT_FALSE(recreated.has_value())
      << "inode " << (new_third.st_ino == old_third.st_ino ? "reused" : "not reused");
  EXPECT_EQ(recreated.error().rule, Rule::kFileType);
  // A shard replaced after open is not the one validated.
  std::error_code error;
  const fs::path shard = root_ / "data" / "00000.safetensors";
  fs::copy_file(shard, scratch_ / "copy", error);
  fs::rename(scratch_ / "copy", shard, error);
  const auto replaced = a.OpenShardForDirectRead(0);
  ASSERT_FALSE(replaced.has_value());
  EXPECT_EQ(replaced.error().rule, Rule::kFileType);
}

TEST_F(ArtifactFilesTest, KeptMetadataIsReadAsListed) {
  const auto opened = Artifact::Open(root_);
  ASSERT_TRUE(opened.has_value());
  const Artifact& a = *opened;
  const fs::path kv = root_ / "meta" / "tiny.kv.gguf";
  const std::string bytes = ReadFile(kv);
  const auto read = a.ReadMetadata("tiny.kv.gguf");
  ASSERT_TRUE(read.has_value()) << read.error().ToString();
  EXPECT_EQ(*read, bytes);
  // Only a listed metadata file, by its name alone.
  for (const char* name : {"nope", "", "../index.json", "meta/tiny.kv.gguf"}) {
    const auto refused = a.ReadMetadata(name);
    ASSERT_FALSE(refused.has_value()) << name;
    EXPECT_EQ(refused.error().rule, Rule::kFileSet) << name;
  }
  // Changed in place since open: its digest differs; its size, if that
  // changed.
  std::string changed = bytes;
  changed.back() = static_cast<char>(changed.back() ^ 1);
  WriteFile(kv, changed);
  auto differs = a.ReadMetadata("tiny.kv.gguf");
  ASSERT_FALSE(differs.has_value());
  EXPECT_EQ(differs.error().rule, Rule::kHash);
  WriteFile(kv, bytes + "x");
  differs = a.ReadMetadata("tiny.kv.gguf");
  ASSERT_FALSE(differs.has_value());
  EXPECT_EQ(differs.error().rule, Rule::kFileSize);
  // A link in its place is never followed.
  std::error_code error;
  fs::remove(kv, error);
  WriteFile(scratch_ / "kv", bytes);
  fs::create_symlink(scratch_ / "kv", kv, error);
  differs = a.ReadMetadata("tiny.kv.gguf");
  ASSERT_FALSE(differs.has_value());
  EXPECT_EQ(differs.error().rule, Rule::kFileType);
}

// ---------------------------------------------------------------- fuzz

// Seeded random mutations of the valid artifacts' documents, built and
// judged by the prototype at build time (artifact_corpus.py fuzz_corpus):
// trees re-hashed and re-listed so that they reach every validation step,
// and raw byte damage, then fixed probes of the two divergences above. Open
// must return the prototype's verdict (ASan and UBSan watch in check:full)
// as the reader reads the artifact: past the prototype's `meta` checks of
// kept GGUF metadata, which the reader does not parse (the corpus records
// that verdict too); and where the reader refuses a string escape, only on
// a document the prototype refuses too, under whatever rule.
TEST(ArtifactCorpusTest, FuzzVerdictsMatchThePrototype) {
  std::istringstream lines(ReadFile(Corpus() / "fuzz.txt"));
  const fs::path scratch = Scratch("fuzz");
  std::map<std::string, fs::path, std::less<>> roots;
  std::map<std::string, std::string, std::less<>> indexes;
  std::string n;
  std::string base;
  std::string oracle;
  std::string unread;
  std::size_t cases = 0;
  std::size_t accepted = 0;
  std::size_t meta = 0;
  std::size_t escapes = 0;
  std::set<std::string> rules;
  while (lines >> n >> base >> oracle >> unread) {
    if (!roots.contains(base)) {
      const fs::path golden = OnlyChild(Corpus() / "golden" / base);
      ASSERT_FALSE(golden.empty()) << base;
      std::error_code error;
      fs::create_directories(scratch / base, error);
      roots[base] = CopyArtifact(golden, scratch / base);
      indexes[base] = ReadFile(golden / "index.json");
    }
    fs::path& root = roots[base];
    const fs::path stem = Corpus() / "fuzz" / n;
    const std::string manifest = ReadFile(fs::path(stem.string() + ".manifest"));
    std::error_code error;
    const fs::path index_file(stem.string() + ".index");
    const std::string index = fs::exists(index_file, error) ? ReadFile(index_file) : indexes[base];
    WriteFile(root / "manifest.json", manifest);
    WriteFile(root / "index.json", index);
    const fs::path renamed = root.parent_path() / Sha256Hex(manifest);
    fs::rename(root, renamed, error);
    ASSERT_FALSE(error) << error.message();
    root = renamed;
    const auto opened = Artifact::Open(root);
    const std::string got = Verdict(opened);
    const bool escape = !opened && opened.error().rule == Rule::kCanonical &&
                        opened.error().reason.starts_with("a string escape");
    if (escape) {
      EXPECT_TRUE(manifest.contains('\\') || index.contains('\\')) << "fuzz case " << n;
      EXPECT_NE(unread, "ok") << "fuzz case " << n << " (" << base
                              << "): the prototype accepts a string escape";
      ++escapes;
    } else {
      EXPECT_EQ(got, unread) << "fuzz case " << n << " (" << base << "): " << Describe(opened)
                             << " (prototype: " << oracle << ")";
    }
    EXPECT_TRUE(oracle == "meta" || unread == oracle) << "fuzz case " << n;
    meta += oracle == "meta" ? 1 : 0;
    accepted += opened.has_value() ? 1 : 0;
    rules.insert(oracle);
    ++cases;
  }
  EXPECT_GE(cases, 808U) << "no corpus at " << Corpus();
  // The mutations reach well past the parser, and both divergences occur.
  EXPECT_GE(accepted, 50U);
  EXPECT_GE(meta, 4U);
  EXPECT_GE(escapes, 4U);
  for (const char* rule : {"json", "schema", "unsupported-version", "bounds", "repr"}) {
    EXPECT_TRUE(rules.contains(rule)) << rule;
  }
}

// ---------------------------------------------------------------- real fixtures

// The v0 fixtures M0 built with the prototype (docs/experiments/artifact-layout),
// where they are installed: on spark, and copied to spark-b. Each view's
// digest is the prototype's (`artifact_corpus.py view`, run on spark on
// 2026-09-27 against the same artifact); counts are the layout study's.
struct Fixture {
  std::string_view id;
  std::string_view what;
  std::size_t groups;
  std::size_t chunks;
  std::size_t placed;  // resources plus expert slices
  std::string_view view_sha256;
};

constexpr std::array kFixtures = {
    Fixture{"b93cdc326ba4f4c1c71da613503ecd848cd2a0caf122214f26e04588a12a9073", "Qwen2.5-0.5B FP16",
            25, 490, 290, "994c1c493672169811834f70c03a5ec4e5257238903be2dca4d351940f58d03a"},
    Fixture{"6e96e499ea9b859ea726327d4f82fb1739cb116ce4e9d3541d1ba67506efb4b4",
            "Qwen2.5-0.5B EXL3 4.0 bpw", 26, 292, 798,
            "950da0fa77ae9a0b09b85e2eb272e5c373b295f4bdd7c34f9dea4a7f64c458ff"},
    Fixture{"00d77caf056c45b6fa05ebdbeff8b38da79d0556a2fd4f1b661e7e21ef9ed83a",
            "Qwen2.5-0.5B EXL3 4.5 bpw", 26, 302, 798,
            "0be0fe43fbcbf24dda8b9a2f618a4f1e83a99f2ad5adbcdfbad15d791adad59c"},
    Fixture{"d673bf28ab0016797432d65c8a888ee7e579c0ad65475e773ebfbdd1bf8e5cfa",
            "Gemma 4 26B-A4B UD-Q4_K_M", 3872, 9059, 8278,
            "ab798d22df74d40024056925a062c86c83911061c7bcf52d93f1b631c93114ca"},
};

TEST(ArtifactFixtureTest, RealArtifactsMatchThePrototype) {
  const std::string home = Env("HOME");
  const fs::path store = fs::path(home) / ".local/share/llmp/artifact-layout-20260922/installed";
  std::size_t found = 0;
  for (const Fixture& f : kFixtures) {
    const fs::path root = store / f.id;
    std::error_code error;
    if (home.empty() || !fs::is_directory(root, error)) {
      continue;
    }
    ++found;
    const auto opened = Artifact::Open(root);
    ASSERT_TRUE(opened.has_value()) << f.what << ": " << Describe(opened);
    const Artifact& a = *opened;
    std::size_t placed = a.resources().size();
    for (const auto& x : a.expert_arrays()) {
      placed += x.count;
    }
    EXPECT_EQ(a.groups().size(), f.groups) << f.what;
    EXPECT_EQ(a.chunk_sha256().size(), f.chunks) << f.what;
    EXPECT_EQ(placed, f.placed) << f.what;
    EXPECT_EQ(Sha256Hex(View(a)), f.view_sha256) << f.what;
  }
  if (found == 0) {
    GTEST_SKIP() << "no v0 fixture under " << store << " (they are on spark and spark-b)";
  }
}

}  // namespace
