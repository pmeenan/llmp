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
// record_to_json.py turns the lines into the JSON files. A digest is bound to
// the selected plans and kernels of the binary that recorded it, so a change
// that legitimately moves bits re-records the file and says so.
//
// The published v1 artifacts' IDs for the tests that take a store: the
// store's v1-ids.json maps each v0 artifact ID to its v1 ID and names them
// (LLMP_TEST_V1_STORE names the store; by default the Sparks' models
// directory's m36-v1-artifacts).

#ifndef LLMP_TESTS_SUPPORT_V1_FROZEN_H_
#define LLMP_TESTS_SUPPORT_V1_FROZEN_H_

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>

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

// The published v1 store: LLMP_TEST_V1_STORE, else the models directory's
// (~/.local/share/llmp/m36-v1-artifacts on the Sparks).
inline std::string V1Store() {
  const char* store = std::getenv("LLMP_TEST_V1_STORE");  // NOLINT(concurrency-mt-unsafe)
  if (store != nullptr) return store;
  const char* home = std::getenv("HOME");  // NOLINT(concurrency-mt-unsafe)
  return home != nullptr ? std::string(home) + "/.local/share/llmp/m36-v1-artifacts" : "";
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
// "qwen2-exl3-4.5"), nullopt for a name it does not have.
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

// The v1 ID the pinned map records for a v0 artifact ID.
inline std::optional<std::string> V1IdFor(std::string_view v0_id) {
  const auto ids = ReadStringMap(V1IdsPath(), "artifacts");
  const auto found = ids.find(std::string(v0_id));
  if (found == ids.end()) return std::nullopt;
  return found->second;
}

}  // namespace llmp::test_support

#endif  // LLMP_TESTS_SUPPORT_V1_FROZEN_H_
