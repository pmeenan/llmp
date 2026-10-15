// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Frozen expectations for the v1 equivalence proofs (M3.6, D-113).
//
// A model's v0 artifact and its v1 import must execute bit for bit alike when
// the selected plan is the same. The proofs first compared the two inside one
// process. Serving becomes v1-only, so the same facts are also recorded as
// SHA-256 digests (full prefill heads, greedy tokens and logit rows,
// complete-state images, the state after a spill and restore) under
// tests/unit/data/v1_frozen/<family>.json, and the v1 arm is asserted
// against them. The v0 arm is still compared and asserted against the same
// digests while a v0 artifact is present, and skipped when it is not.
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
// store's v1-ids.json maps each v0 artifact ID to its v1 ID
// (LLMP_TEST_V1_STORE names the store).

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

// The store named by LLMP_TEST_V1_STORE, empty when unset.
inline std::string V1Store() {
  const char* store = std::getenv("LLMP_TEST_V1_STORE");  // NOLINT(concurrency-mt-unsafe)
  return store != nullptr ? store : "";
}
// The v1 ID the store's v1-ids.json records for a v0 artifact ID.
inline std::optional<std::string> V1IdFor(std::string_view v0_id) {
  const auto store = V1Store();
  if (store.empty()) return std::nullopt;
  const auto ids = ReadStringMap(store + "/v1-ids.json", "artifacts");
  const auto found = ids.find(std::string(v0_id));
  if (found == ids.end()) return std::nullopt;
  return found->second;
}

}  // namespace llmp::test_support

#endif  // LLMP_TESTS_SUPPORT_V1_FROZEN_H_
