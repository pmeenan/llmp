// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Compiled, declarative source normalization. These tables describe input
// authority, not callbacks or permission to execute a checkpoint.
#ifndef LLMP_IMPORT_SOURCE_MAPPING_H_
#define LLMP_IMPORT_SOURCE_MAPPING_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string_view>

#include "import/error.h"
#include "spec/mapping.h"

namespace llmp::import {

struct TensorAlias {
  std::string_view source{};
  std::string_view canonical{};  // an existing TensorRule pattern
};
struct KeyAlias {
  std::string_view source{};
  std::string_view canonical{};  // an existing selected source key
};
struct ScalarRequirement {
  // Refuse this integer by default; require_equal instead admits only it.
  std::string_view key{};
  std::int64_t reject_equal = 0;
  bool require_equal = false;
  std::string_view equals_key{};  // require equality with another selected scalar
  bool positive = false;
  // A legacy ambiguity refusal is waived only by this explicit semantic key.
  std::string_view unless_present{};
  bool optional = false;
};
struct TensorPresenceRequirement {
  std::string_view flag{}, tensor{};
  bool when = false;
};
struct ListRequirement {
  std::string_view key{};
  std::size_t size = 0;
  std::span<const std::int64_t> exact{};  // absent: retain the source values
  // Nonzero: the list may hold from `min_size` to `size` entries (trailing
  // entries describe layers the source does not hold), equal to `exact`'s
  // first entries where it is given.
  std::size_t min_size = 0;
  // The key may be absent (the mapping resolves its parameter otherwise).
  bool optional = false;
  bool per_layer = false;
  std::string_view extra_layers_key{};
  // Some converters append type labels for unimported layers without a depth key.
  bool allow_unstated_extra = false;
};
struct TensorShapeRequirement {
  std::string_view name{};
  std::span<const std::uint64_t> shape{};
};
struct RowIndexRequirement {
  std::string_view offsets{}, counts{}, tensor{};
  bool operator==(const RowIndexRequirement&) const = default;
};
struct AuxiliaryNamespace {
  std::string_view prefix{};  // literal: dot-terminated, or one whole key; metadata only
  std::string_view reason{};
  // A whole key's integer that must equal a source tensor's extent along
  // `equals_dim` (a vocabulary size its token table states): an inconsistent
  // source is refused. Empty: unchecked.
  std::string_view equals_tensor{};
  std::uint8_t equals_dim = 0;
};
enum class RouteOperation : std::uint8_t { kCopyPart };
struct SourceMapping {
  std::string_view container{};
  std::uint16_t version = 1;
  const spec::Mapping* architecture = nullptr;
  std::span<const TensorAlias> tensors{};
  std::span<const KeyAlias> keys{};
  std::span<const AuxiliaryNamespace> auxiliary{};
  std::span<const ScalarRequirement> requirements{};
  RouteOperation route = RouteOperation::kCopyPart;
  std::span<const ListRequirement> lists{};
  std::span<const format::Encoding> encodings{};
  // Compiled bounded side/mixing and numeric-list component domain.
  bool extended_components = false;
  std::span<const TensorShapeRequirement> shapes{};
  std::span<const RowIndexRequirement> row_indices{};
  // Unknown metadata in these math-bearing namespaces cannot be ignored.
  std::span<const std::string_view> semantic{};
  std::span<const std::string_view> ignored_tensors{};
  // These explicit JSON nulls mean that an optional semantic feature is absent.
  std::span<const std::string_view> null_semantics{};
  // A text pipeline may explicitly exclude other components of its container.
  std::span<const std::string_view> ignored_tensor_prefixes{};
  std::span<const TensorPresenceRequirement> tensor_presence{};
};

const SourceMapping* FindSourceMapping(std::string_view container, std::string_view architecture);
// Allocation-free validation of the bounded normalization schema, including
// aliases' whole-pattern overlap. Full spec::ValidateMapping runs after funding.
std::expected<void, Error> ValidateSourceMapping(const SourceMapping& mapping);

namespace normalization_detail {
inline constexpr std::uint32_t kLayerLimit = 1024;
inline constexpr std::size_t kRuleLimit = 64;
// Bits for the accepted source scalar and layer-list alternatives. Int and Real intersect
// when one key supplies both; a bool is never an integer.
inline constexpr std::uint8_t kInteger = 1, kReal = 2, kFlag = 4, kText = 8, kIntegers = 16,
                              kReals = 32;
std::uint8_t KeyTypes(const spec::Mapping& mapping, std::string_view key);
bool Pattern(std::string_view pattern);
struct NameMatch {
  std::string_view digits{};
};
std::optional<NameMatch> Match(std::string_view pattern, std::string_view name,
                               std::uint32_t layers);
}  // namespace normalization_detail
}  // namespace llmp::import
#endif  // LLMP_IMPORT_SOURCE_MAPPING_H_
