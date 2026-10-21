// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The v1 artifact reader (src/artifact): its strict JSON and hex digests, read
// planning on hand-built layouts, and Artifact::Open against filesystem edge
// cases (links, special files, unreadable directories, sparse files, direct
// reads, kept metadata).

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
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "artifact/error.h"
#include "artifact/json.h"
#include "artifact/layout.h"
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

std::string Verdict(const std::expected<Artifact, Error>& opened) {
  return opened ? std::string("ok") : std::string(RuleName(opened.error().rule));
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

// Builds a small valid 3-shard v1 artifact under `into` and returns its path.
fs::path WriteTinyV1Artifact(const fs::path& into) {
  const auto make_shard = [](std::string_view entries, std::size_t data_bytes) {
    std::string text(8, '\0');
    text += R"({"__metadata__":{"format":"llmp-shard","format_version":"1"},)";
    text += entries;
    text += '}';
    text.resize(4096, ' ');
    const std::uint64_t length = 4096 - 8;
    for (std::size_t k = 0; k < 8; ++k) {
      text[k] = static_cast<char>((length >> (8U * k)) & 0xFFU);
    }
    const std::string header_sha = Sha256Hex(text);
    text.append(data_bytes, '\0');
    return std::pair(std::move(text), header_sha);
  };
  const auto [s0_bytes, s0_sha] = make_shard(
      R"("token_embd.weight":{"dtype":"F32","shape":[8,128],"data_offsets":[0,4096]})", 4096);
  const auto [s1_bytes, s1_sha] = make_shard(
      R"("output_norm.weight":{"dtype":"F32","shape":[1024],"data_offsets":[0,4096]})", 4096);
  const auto [s2_bytes, s2_sha] = make_shard(
      R"("blk.1.ffn_down_exps.weight#0":{"dtype":"F32","shape":[4,256],"data_offsets":[0,4096]},"blk.1.ffn_down_exps.weight#1":{"dtype":"F32","shape":[4,256],"data_offsets":[4096,8192]})",
      8192);
  const std::string zero_chunk_sha = Sha256Hex(std::string(4096, '\0'));

  const auto f32_format = [](std::initializer_list<std::int64_t> ne) {
    J f = J::Of(json::Kind::kObject);
    f["encoding"] = J::Str("f32");
    f["layout"] = J::Str("dense");
    f["ne"] = J::Of(json::Kind::kArray);
    for (std::int64_t d : ne) {
      f["ne"].items.push_back(J::Int(d));
    }
    f["params"] = J::Of(json::Kind::kObject);
    f["stacked"] = J::Int(1);
    return f;
  };
  const auto make_group = [](std::string_view kind, std::optional<std::int64_t> layer,
                             std::optional<std::int64_t> expert, std::int64_t shard,
                             std::int64_t offset, std::int64_t first_chunk) {
    J g = J::Of(json::Kind::kObject);
    g["expert"] = expert ? J::Int(*expert) : J::Of(json::Kind::kNull);
    g["first_chunk"] = J::Int(first_chunk);
    g["kind"] = J::Str(std::string(kind));
    g["layer"] = layer ? J::Int(*layer) : J::Of(json::Kind::kNull);
    g["offset"] = J::Int(offset);
    g["shard"] = J::Int(shard);
    g["stored_bytes"] = J::Int(4096);
    g["used_bytes"] = J::Int(4096);
    return g;
  };
  const auto make_shard_entry = [](std::string_view path, std::int64_t data_bytes,
                                   const std::string& header_sha) {
    J s = J::Of(json::Kind::kObject);
    s["data_bytes"] = J::Int(data_bytes);
    s["data_offset"] = J::Int(4096);
    s["header_sha256"] = J::Str(header_sha);
    s["path"] = J::Str(std::string(path));
    return s;
  };

  J index = J::Of(json::Kind::kObject);
  index["chunk_bytes"] = J::Int(static_cast<std::int64_t>(artifact::kChunkBytes));
  index["chunk_sha256"] = J::Of(json::Kind::kArray);
  for (int i = 0; i < 4; ++i) {
    index["chunk_sha256"].items.push_back(J::Str(zero_chunk_sha));
  }
  J arr = J::Of(json::Kind::kObject);
  arr["count"] = J::Int(2);
  arr["first_group"] = J::Int(2);
  arr["format"] = f32_format({256, 4});
  arr["group_offset"] = J::Int(0);
  arr["layer"] = J::Int(1);
  arr["name"] = J::Str("blk.1.ffn_down_exps.weight");
  arr["part"] = J::Str("data");
  arr["readable_bytes"] = J::Int(4096);
  arr["slice_bytes"] = J::Int(4096);
  index["expert_arrays"] = J::Of(json::Kind::kArray);
  index["expert_arrays"].items.push_back(std::move(arr));
  index["file_alignment"] = J::Int(4096);
  index["format"] = J::Str("llmp-index");
  index["format_version"] = J::Int(1);
  index["groups"] = J::Of(json::Kind::kArray);
  index["groups"].items.push_back(make_group("table", std::nullopt, std::nullopt, 0, 0, 0));
  index["groups"].items.push_back(make_group("head", std::nullopt, std::nullopt, 1, 0, 1));
  index["groups"].items.push_back(make_group("expert", 1, 0, 2, 0, 2));
  index["groups"].items.push_back(make_group("expert", 1, 1, 2, 4096, 3));
  index["member_alignment"] = J::Int(256);

  J r0 = J::Of(json::Kind::kObject);
  r0["access"] = J::Str("rows");
  r0["bytes"] = J::Int(4096);
  r0["format"] = f32_format({128, 8});
  r0["group"] = J::Int(0);
  r0["name"] = J::Str("token_embd.weight");
  r0["offset"] = J::Int(0);
  r0["part"] = J::Str("data");
  r0["readable_bytes"] = J::Int(4096);
  r0["roles"] = J::Of(json::Kind::kArray);
  r0["roles"].items = {J::Str("token_embd.weight"), J::Str("output.weight")};

  J r1 = J::Of(json::Kind::kObject);
  r1["bytes"] = J::Int(4096);
  r1["format"] = f32_format({1024});
  r1["group"] = J::Int(1);
  r1["name"] = J::Str("output_norm.weight");
  r1["offset"] = J::Int(0);
  r1["part"] = J::Str("data");
  r1["readable_bytes"] = J::Int(4096);
  r1["roles"] = J::Of(json::Kind::kArray);
  r1["roles"].items = {J::Str("output_norm.weight")};

  index["resources"] = J::Of(json::Kind::kArray);
  index["resources"].items = {std::move(r0), std::move(r1)};
  index["shards"] = J::Of(json::Kind::kArray);
  index["shards"].items = {
      make_shard_entry("data/00000.safetensors", 4096, s0_sha),
      make_shard_entry("data/00001.safetensors", 4096, s1_sha),
      make_shard_entry("data/00002.safetensors", 8192, s2_sha),
  };

  const fs::path stage = into / "staging";
  std::error_code error;
  fs::create_directories(stage / "data", error);
  fs::create_directories(stage / "meta", error);
  WriteFile(stage / "index.json", index.Canonical());
  WriteFile(stage / "data" / "00000.safetensors", s0_bytes);
  WriteFile(stage / "data" / "00001.safetensors", s1_bytes);
  WriteFile(stage / "data" / "00002.safetensors", s2_bytes);
  const std::string kv_bytes = "GGUF\x03\x00\x00\x00";
  WriteFile(stage / "meta" / "tiny.kv.gguf", kv_bytes);

  J manifest = J::Of(json::Kind::kObject);
  manifest["experimental"] = J::Of(json::Kind::kTrue);
  manifest["files"] = J::Of(json::Kind::kArray);
  AddFile(manifest, "data/00000.safetensors", "shard", s0_bytes.size());
  AddFile(manifest, "data/00001.safetensors", "shard", s1_bytes.size());
  AddFile(manifest, "data/00002.safetensors", "shard", s2_bytes.size());
  AddFile(manifest, "index.json", "index", 0);
  AddFile(manifest, "meta/tiny.kv.gguf", "source-metadata", kv_bytes.size());
  manifest["format"] = J::Str("llmp-artifact");
  manifest["format_version"] = J::Int(1);
  J importer = J::Of(json::Kind::kObject);
  importer["build"] = J::Str("unknown");
  importer["name"] = J::Str("llmp-import");
  importer["options_sha256"] = J::Str(Sha256Hex(""));
  importer["version"] = J::Str("0.0.0");
  manifest["importer"] = std::move(importer);
  J layout = J::Of(json::Kind::kObject);
  layout["byte_order"] = J::Str("little");
  layout["chunk_bytes"] = J::Int(static_cast<std::int64_t>(artifact::kChunkBytes));
  layout["container"] = J::Str("safetensors");
  layout["file_alignment"] = J::Int(4096);
  layout["member_alignment"] = J::Int(256);
  layout["profile"] = J::Str("spark");
  layout["profile_version"] = J::Int(1);
  manifest["layout"] = std::move(layout);
  J model = J::Of(json::Kind::kObject);
  model["architecture"] = J::Str("llama");
  model["container"] = J::Str("gguf");
  model["expert_count"] = J::Int(2);
  manifest["model"] = std::move(model);
  manifest["source"] = J::Of(json::Kind::kArray);
  J src = J::Of(json::Kind::kObject);
  src["bytes"] = J::Int(16);
  src["name"] = J::Str("tiny.gguf");
  src["sha256"] = J::Str(Sha256Hex("tiny-gguf-source"));
  manifest["source"].items.push_back(std::move(src));
  J t_slice = J::Of(json::Kind::kObject);
  t_slice["count"] = J::Int(2);
  t_slice["kind"] = J::Str("expert-slice");
  t_slice["tensor"] = J::Str("blk.1.ffn_down_exps.weight");
  J t_dedupe = J::Of(json::Kind::kObject);
  t_dedupe["kind"] = J::Str("dedupe-identical");
  t_dedupe["resource"] = J::Str("token_embd.weight");
  t_dedupe["role"] = J::Str("output.weight");
  t_dedupe["sha256"] = J::Str(zero_chunk_sha);
  manifest["transformations"] = J::Of(json::Kind::kArray);
  manifest["transformations"].items = {std::move(t_slice), std::move(t_dedupe)};
  WriteFile(stage / "manifest.json", manifest.Canonical());
  return Relist(stage, [](J&) {});
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

TEST(ArtifactJsonTest, ExtendedFileNamesAndStableBuildIdentity) {
  const std::string unknown = R"({"files":[],"importer":{"build":"unknown","name":"llmp-import"}})"
                              "\n";
  const std::string other =
      R"({"files":[],"importer":{"build":"40bf23e3.dirty","name":"llmp-import"}})"
      "\n";
  const auto old = json::Parse(unknown, true);
  const auto current = json::Parse(other, true);
  ASSERT_TRUE(old && current);
  EXPECT_EQ(llmp::artifact::schema::ManifestIdentity(old->root()),
            llmp::artifact::schema::ManifestIdentity(current->root()));
  EXPECT_EQ(llmp::base::ToHex(llmp::artifact::schema::ManifestIdentity(old->root())),
            Sha256Hex(unknown));
  const std::string names = R"(["model \"café\".gguf","tokenizer_2/tokenizer.json"])"
                            "\n";
  const auto parsed = json::Parse(names, true);
  ASSERT_TRUE(parsed) << parsed.error().ToString();
  EXPECT_TRUE(parsed->canonical());
  std::string canonical;
  json::Serialize(parsed->root(), canonical);
  EXPECT_EQ(canonical + '\n', names);
  EXPECT_FALSE(json::Parse(names));  // v0 remains the original strict subset.
  EXPECT_FALSE(json::Parse(R"(["\ud800"])"
                           "\n",
                           true));
  EXPECT_FALSE(json::Parse(R"({"a":1,"\u0061":2})"
                           "\n",
                           true));
  EXPECT_FALSE(json::Parse(std::string(10, '[') + "0" + std::string(10, ']'), true));
}

TEST(ArtifactJsonTest, CanonicalDocumentsRoundTrip) {
  // Any document the parser calls canonical re-serializes to its own bytes.
  const fs::path root = WriteTinyV1Artifact(Scratch("roundtrip"));
  const std::string index = ReadFile(root / "index.json");
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

// ---------------------------------------------------------------- the directory

class ArtifactFilesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    scratch_ = Scratch(::testing::UnitTest::GetInstance()->current_test_info()->name());
    root_ = WriteTinyV1Artifact(scratch_);
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

TEST_F(ArtifactFilesTest, RefusesV0FormatVersionAndProfile) {
  root_ = Relist(root_, [](J& m) { m["format"] = J::Str("jitllm-artifact"); });
  EXPECT_EQ(Open(root_), "format");
  root_ = Relist(root_, [](J& m) {
    m["format"] = J::Str("llmp-artifact");
    m["format_version"] = J::Int(0);
  });
  EXPECT_EQ(Open(root_), "unsupported-version");
  root_ = Relist(root_, [](J& m) {
    m["format_version"] = J::Int(1);
    m["layout"]["profile"] = J::Str("spark-v0");
    m["layout"]["profile_version"] = J::Int(0);
  });
  EXPECT_EQ(Open(root_), "unsupported-profile");
}

TEST_F(ArtifactFilesTest, LinksInsideAndSpecialFilesAreRefused) {
  std::error_code error;
  // The root through a link is the directory it leads to (an artifact
  // kept on another drive; D-056 as amended on 2026-10-09), named by the
  // link, and its shards reopen through it.
  const fs::path store = scratch_ / "store";
  fs::create_directory(store, error);
  fs::create_directory_symlink(root_, store / root_.filename(), error);
  ASSERT_FALSE(error) << error.message();
  {
    const auto linked = Artifact::Open(store / root_.filename());
    ASSERT_TRUE(linked.has_value()) << linked.error().ToString();
    EXPECT_TRUE(linked->OpenShardForDirectRead(0).has_value());
  }
  fs::create_directory_symlink(root_, scratch_ / "link", error);
  EXPECT_EQ(Open(scratch_ / "link"), "identity");  // named by the link, as any root
  EXPECT_EQ(Open(scratch_ / "link", {.expected_id = root_.filename().string()}), "ok");
  // A link to a link to nothing is not a directory.
  fs::create_directory_symlink(scratch_ / "absent", scratch_ / "dangling", error);
  EXPECT_EQ(Open(scratch_ / "dangling", {.expected_id = root_.filename().string()}), "file-type");
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
  // A hard-linked file is read as any file (a deduplicated store), and
  // its shard reopens for direct reads.
  const fs::path shard = root_ / "data" / "00000.safetensors";
  fs::create_hard_link(shard, scratch_ / "second-name", error);
  ASSERT_FALSE(error) << error.message();
  {
    const auto linked = Artifact::Open(root_);
    ASSERT_TRUE(linked.has_value()) << linked.error().ToString();
    EXPECT_TRUE(linked->OpenShardForDirectRead(0).has_value());
  }
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
  EXPECT_EQ(Open(shard, {.expected_id = root_.filename().string()}), "file-type");
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

// Who could change the store is the filesystem's business (D-073 as
// amended on 2026-10-09): modes are not the reader's.
TEST_F(ArtifactFilesTest, ModesAreNotTheReadersBusiness) {
  const fs::path shard = root_ / "data" / "00001.safetensors";
  std::error_code error;
  fs::permissions(shard, fs::perms::others_write, fs::perm_options::add, error);
  fs::permissions(root_ / "meta", fs::perms::others_write, fs::perm_options::add, error);
  EXPECT_EQ(Open(root_), "ok");
  fs::permissions(shard, fs::perms::others_write, fs::perm_options::remove, error);
  fs::permissions(root_ / "meta", fs::perms::others_write, fs::perm_options::remove, error);
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
  // A shard is reopened whenever its model becomes active again, perhaps
  // long after Open (engine/paged_weights.h), and checked for what the
  // reads need, not its identity (D-056 as amended on 2026-10-10): one
  // whose status changed (its mode) or that was replaced by a copy of the
  // same size still opens.
  const fs::path first = root_ / "data" / "00000.safetensors";
  fs::permissions(first, fs::perms::owner_read | fs::perms::group_read, fs::perm_options::replace);
  EXPECT_TRUE(a.OpenShardForDirectRead(0).has_value());
  fs::permissions(first, fs::perms::owner_read | fs::perms::owner_write, fs::perm_options::replace);
  std::error_code error;
  fs::copy_file(first, scratch_ / "copy", error);
  fs::rename(scratch_ / "copy", first, error);
  ASSERT_FALSE(error) << error.message();
  EXPECT_TRUE(a.OpenShardForDirectRead(0).has_value());
  // One of another size than the installed manifest lists is refused,
  // longer or shorter.
  const fs::path second = root_ / "data" / "00001.safetensors";
  const std::string content = ReadFile(second);
  WriteFile(second, content + std::string(4096, '\0'));
  const auto longer = a.OpenShardForDirectRead(1);
  ASSERT_FALSE(longer.has_value());
  EXPECT_EQ(longer.error().rule, Rule::kFileSize);
  WriteFile(second, content.substr(0, content.size() - 1));
  const auto shorter = a.OpenShardForDirectRead(1);
  ASSERT_FALSE(shorter.has_value());
  EXPECT_EQ(shorter.error().rule, Rule::kFileSize);
  WriteFile(second, content);
  EXPECT_TRUE(a.OpenShardForDirectRead(1).has_value());
  // Gone: refused, as any file the open cannot find.
  const fs::path third = root_ / "data" / "00002.safetensors";
  fs::rename(third, scratch_ / "third", error);
  ASSERT_FALSE(error) << error.message();
  EXPECT_FALSE(a.OpenShardForDirectRead(2).has_value());
  fs::rename(scratch_ / "third", third, error);
  EXPECT_TRUE(a.OpenShardForDirectRead(2).has_value());
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

}  // namespace
