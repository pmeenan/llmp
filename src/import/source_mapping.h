// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Declarative source-normalization tables (M3.6 N2; D-109, D-121). A table
// projects a reader's raw checkpoint inventory into the canonical tensor and
// metadata namespace consumed by spec::Mapping without family-specific C++
// branches. Compiled tables are validated as untrusted data before use.

#ifndef LLMP_IMPORT_SOURCE_MAPPING_H_
#define LLMP_IMPORT_SOURCE_MAPPING_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string_view>

#include "format/encoding.h"
#include "import/checkpoint.h"
#include "spec/mapping.h"

namespace llmp::import {

enum class RouteOperation : std::uint8_t { kCopyPart };

using spec::AuxiliaryNamespace;
using spec::KeyAlias;
using spec::ListRequirement;
using spec::RowIndexRequirement;
using spec::ScalarRequirement;
using spec::TensorAlias;
using spec::TensorPresenceRequirement;
using spec::TensorShapeRequirement;

struct SourceMapping {
  std::string_view container{};
  std::uint16_t version = 1;
  const spec::Mapping* architecture = nullptr;
  RouteOperation route = RouteOperation::kCopyPart;
  std::span<const TensorAlias> tensors{};
  std::span<const KeyAlias> keys{};
  std::span<const AuxiliaryNamespace> auxiliary{};
  std::span<const ScalarRequirement> requirements{};
  std::span<const ListRequirement> lists{};
  std::span<const format::Encoding> encodings{};  // empty: any reader-supported encoding
  // Allows side-input, stream-mixing, per-layer real-list and integer-list
  // rules for registered complex architectures.
  bool extended_components = false;
  std::span<const TensorShapeRequirement> shapes{};
  std::span<const RowIndexRequirement> row_indices{};
  // Additional unprefixed source keys or glob patterns ("rope_*") that are
  // model-math semantics in this container when not selected or aliased.
  std::span<const std::string_view> semantic{};
  // Redundant source tensors (such as precomputed rotary frequencies) skipped
  // deliberately rather than routed into the artifact.
  std::span<const std::string_view> ignored_tensors{};
  // Keys that may appear in config.json with a JSON null value meaning "absent".
  std::span<const std::string_view> null_semantics{};
  // Source tensor name prefixes outside the target pipeline (for example vision
  // towers in a multimodal checkpoint imported for its text model).
  std::span<const std::string_view> ignored_tensor_prefixes{};
  // Boolean metadata flags that require a canonical tensor when equal to `when`.
  std::span<const TensorPresenceRequirement> tensor_presence{};
};

const SourceMapping* FindSourceMapping(std::string_view container, std::string_view architecture,
                                       std::string_view component = {});
std::expected<void, Error> ValidateSourceMapping(const SourceMapping& table);

namespace normalization_detail {
inline constexpr std::size_t kRuleLimit = 512;
inline constexpr std::uint32_t kLayerLimit = 1024;
enum ValueMask : std::uint8_t {
  kInteger = 1U << 0U,
  kReal = 1U << 1U,
  kFlag = 1U << 2U,
  kText = 1U << 3U,
  kIntegers = 1U << 4U,
  kReals = 1U << 5U
};
struct NameMatch {
  std::string_view digits{};
  std::string_view index_digits{};
  std::optional<std::uint32_t> index{};
  bool is_expert = false;
  bool is_shard = false;
};
std::uint8_t KeyTypes(const spec::Mapping& m, std::string_view key, bool allow_unprefixed = false);
bool Pattern(std::string_view pattern);
std::optional<NameMatch> Match(std::string_view pattern, std::string_view name,
                               std::uint32_t layers);
}  // namespace normalization_detail

}  // namespace llmp::import

#endif  // LLMP_IMPORT_SOURCE_MAPPING_H_
