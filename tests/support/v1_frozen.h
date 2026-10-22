// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Frozen expectations for the v1 equivalence proofs (M3.6, D-113).
//
// A model's v0 artifact and its v1 import must execute bit for bit alike when
// the selected plan is the same. The proofs first compared the two inside one
// process, and the facts were recorded as SHA-256 digests (full prefill
// heads, greedy tokens and logit rows, complete-state images, the state
// after a spill and restore) under tests/unit/data/v1_frozen/<family>.json.
// Serving is v1-only (D-118): the v1 artifacts are asserted against them,
// the regression oracle that outlives the v0 path.
//
// Reading: LLMP_TEST_DATA/v1_frozen/<family>.json, a flat object of strings
// (or LLMP_TEST_V1_FROZEN_DIR/<family>.json). Recording: with
// LLMP_TEST_V1_RECORD=<file> set, Expect appends "<family> <key> <value>"
// lines to the file instead of asserting; tests/unit/data/v1_frozen/
// record_to_json.py turns the lines into the JSON files (or runs CTest and
// updates the JSON files in one command:
//   python3 tests/unit/data/v1_frozen/record_to_json.py \
//       --ctest 'GemmaNativeV1ServingGpu' --provenance '...'
// ). A digest is bound to the selected plans and kernels of the binary that
// recorded it, so a change that legitimately moves bits re-records the file
// and says so; same-binary invariants (finite logits, greedy token argmax,
// spill/restore state equality) are asserted alongside the frozen digests.
//
// The published v1 artifacts' IDs for the tests that take a store: the
// store's v1-ids.json maps each v0 artifact ID to its v1 ID and names them
// (LLMP_TEST_V1_STORE names the store; by default the Sparks' models
// directory's `artifacts`, falling back to `m36-v1-artifacts`).

#ifndef LLMP_TESTS_SUPPORT_V1_FROZEN_H_
#define LLMP_TESTS_SUPPORT_V1_FROZEN_H_

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>

#include "base/bytes.h"
#include "base/json.h"
#include "base/sha256.h"

namespace llmp::test_support {

inline std::string HexOfBytes(std::span<const std::byte> bytes) {
  base::Sha256 hash;
  hash.Update(bytes);
  return base::ToHex(hash.Finish());
}
// The digest of the rows' raw bytes, in order.
inline std::string HexOfFloats(std::span<const float> values) {
  return HexOfBytes(std::as_bytes(values));
}
inline std::string HexOfInts(std::span<const std::int32_t> values) {
  return HexOfBytes(std::as_bytes(values));
}
// One digest over several float rows (a generation's logit rows), each row's
// length folded in so a split cannot alias a concatenation.
template <class Rows>
std::string HexOfRows(const Rows& rows) {
  base::Sha256 hash;
  for (const auto& row : rows) {
    const std::uint64_t n = row.size();
    hash.Update(std::as_bytes(std::span(&n, 1)));
    hash.Update(std::as_bytes(std::span<const float>(row.data(), row.size())));
  }
  return base::ToHex(hash.Finish());
}
inline std::string JoinTokens(std::span<const std::int32_t> tokens) {
  std::string out;
  for (const auto token : tokens) {
    if (!out.empty()) out += ',';
    out += std::to_string(token);
  }
  return out;
}

inline std::optional<std::string> ReadWhole(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return std::nullopt;
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
}

// A flat JSON object of strings, nothing else.
inline std::map<std::string, std::string> ReadStringMap(const std::string& path,
                                                        const char* object = nullptr) {
  std::map<std::string, std::string> out;
  const auto bytes = ReadWhole(path);
  if (!bytes) return out;
  auto parsed = base::json::Parse(*bytes);
  if (!parsed) return out;
  auto root = parsed->root();
  if (object != nullptr) {
    const auto inner = root.find(object);
    if (!inner || !inner->is_object()) return out;
    root = *inner;
  }
  if (!root.is_object()) return out;
  for (std::size_t i = 0; i < root.size(); ++i)
    if (root.member(i).is_string())
      out.emplace(std::string(root.key(i)), std::string(root.member(i).string()));
  return out;
}

class FrozenDigests {
 public:
  explicit FrozenDigests(std::string family) : family_(std::move(family)) {
    const char* record = std::getenv("LLMP_TEST_V1_RECORD");  // NOLINT(concurrency-mt-unsafe)
    if (record != nullptr && record[0] != '\0') {
      record_ = record;
      return;
    }
    const char* dir = std::getenv("LLMP_TEST_V1_FROZEN_DIR");  // NOLINT(concurrency-mt-unsafe)
    std::string path;
    if (dir != nullptr) {
      path = std::string(dir) + "/" + family_ + ".json";
    } else {
      const char* data = std::getenv("LLMP_TEST_DATA");  // NOLINT(concurrency-mt-unsafe)
      path = std::string(data != nullptr ? data : "") + "/v1_frozen/" + family_ + ".json";
    }
    digests_ = ReadStringMap(path, "digests");
    loaded_ = !digests_.empty();
    path_ = std::move(path);
  }
  bool recording() const { return record_.has_value(); }
  bool loaded() const { return loaded_; }
  // Asserts `value` equals the frozen digest of `key`, or records it.
  void Expect(const std::string& key, const std::string& value) {
    if (record_) {
      std::ofstream out(*record_, std::ios::app);
      out << family_ << ' ' << key << ' ' << value << '\n';
      return;
    }
    const auto found = digests_.find(key);
    if (found == digests_.end()) {
      ADD_FAILURE() << "no frozen digest '" << key << "' for " << family_ << " in " << path_;
      return;
    }
    EXPECT_EQ(value, found->second) << family_ << " " << key << " differs from its frozen digest";
  }

 private:
  std::string family_;
  std::optional<std::string> record_;
  std::map<std::string, std::string> digests_;
  std::string path_;
  bool loaded_ = false;
};

// The standing models directory: LLMP_TEST_MODELS, else $HOME/.local/share/llmp.
inline std::string StandingModelsDir() {
  const char* models = std::getenv("LLMP_TEST_MODELS");  // NOLINT(concurrency-mt-unsafe)
  if (models != nullptr && models[0] != '\0') return models;
  const char* home = std::getenv("HOME");  // NOLINT(concurrency-mt-unsafe)
  return home != nullptr ? std::string(home) + "/.local/share/llmp" : "";
}

// Standing Qwen2 FP16 and EXL3 trajectory inputs and sealed oracles on the Sparks.
inline std::string Qwen2OraclePath(std::string_view file) {
  const auto base = StandingModelsDir();
  if (base.empty()) return "";
  const std::filesystem::path raw =
      std::filesystem::path(base) / "references" / "qwen2-oracles" / file;
  std::error_code ec;
  const auto canon = std::filesystem::canonical(raw, ec);
  return ec ? raw.string() : canon.string();
}

// The published v1 store: LLMP_TEST_V1_STORE, else the models directory's
// `u2-v1-artifacts` store (falling back to `artifacts` and `m36-v1-artifacts`).
inline std::string V1Store() {
  const char* store = std::getenv("LLMP_TEST_V1_STORE");  // NOLINT(concurrency-mt-unsafe)
  if (store != nullptr && store[0] != '\0') return store;
  const auto base = StandingModelsDir();
  if (base.empty()) return "";
  std::error_code ec;
  const std::string u2 = base + "/u2-v1-artifacts";
  if (std::filesystem::is_directory(u2, ec)) return u2;
  const std::string canonical = base + "/artifacts";
  if (std::filesystem::is_directory(canonical, ec)) return canonical;
  const std::string legacy = base + "/m36-v1-artifacts";
  if (std::filesystem::is_directory(legacy, ec)) return legacy;
  return canonical;
}
// The published v1 IDs, pinned in-tree: tests/unit/data/v1_frozen/v1-ids.json
// (a copy of the store's), read from LLMP_TEST_DATA (else ctest's working
// directory's data/). A models test that names an artifact fails, rather
// than skips, when the store does not hold it: its ID never depends on the
// store being there.
inline std::string V1IdsPath() {
  const char* data = std::getenv("LLMP_TEST_DATA");  // NOLINT(concurrency-mt-unsafe)
  return std::string(data != nullptr ? data : "data") + "/v1_frozen/v1-ids.json";
}

// The v1 ID the pinned map names `name` (its "names" map: "gemma2",
// "gemma3", "gemma26", "gemma26-assistant", "gemma31", "gemma31-assistant",
// "qwen38-gguf", "qwen38-modelopt", "qwen38-modelopt-mtp", "qwen38-mlx",
// "deepseek", "deepseek-iq2xxs", "dspark", "qwen2-fp16", "qwen2-exl3-4.0",
// "qwen2-exl3-4.5", "llama32-1b-q8_0", "llama32-1b-q4_k_m", and the Qwen-Image composition
// "qwen-image" with its "qwen-image-text-encoder", "qwen-image-transformer" and "qwen-image-vae"),
// nullopt for a name it does not have.
inline std::optional<std::string> V1IdNamed(std::string_view name) {
  const auto path = V1IdsPath();
  const auto names = ReadStringMap(path, "names");
  const auto ids = ReadStringMap(path, "artifacts");
  for (const auto& [key, named] : names) {
    if (named != name) continue;
    const auto found = ids.find(key);
    if (found == ids.end()) return std::nullopt;
    return found->second;
  }
  return std::nullopt;
}

namespace v1_store {

inline constexpr std::string_view kQwen38ModelOptTarget =
    "02354af162f30b6ec6a117d5311da7d642c5be3128867750be55eeb7b6cd38b9";
inline constexpr std::string_view kQwen38Gguf =
    "f60d801a1695079ba4cd9c17d32967315bf77fc26580c1f8245b37f95d03d027";
inline constexpr std::string_view kQwen38ModelOptDrafter =
    "2cb4cdb8d4c8bba3bf7a258d56edbae1b456d9268c238948f00210f7b1e9d0ef";
inline constexpr std::string_view kDeepSeekV4Target =
    "d2046349e81da7019c9d787c2eb92521b7a3f4ca9d09c398448da1a99f8eef3f";
inline constexpr std::string_view kDeepSeekV4Drafter =
    "6b1b9dcafe85e07e37a0b14007280da85cd0406158eaf73a658bbaf1574eec41";
inline constexpr std::string_view kQwenImageComposition =
    "62497918eae947b8d4327bf8e0771318118d581c59c2c0574208345ed7cc6518";

}  // namespace v1_store

}  // namespace llmp::test_support

#endif  // LLMP_TESTS_SUPPORT_V1_FROZEN_H_
