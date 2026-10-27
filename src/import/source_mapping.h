// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Import selection and validation of spec::Mapping (D-109, D-121). The
// authoritative mapping describes both source dialect and component structure.
// Compiled declarations are validated before interpreting checkpoint data.

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

using spec::AuxiliaryNamespace;
using spec::KeyAlias;
using spec::ListRequirement;
using spec::RowIndexRequirement;
using spec::ScalarRequirement;
using spec::TensorAlias;
using spec::TensorPresenceRequirement;
using spec::TensorShapeRequirement;

// Select directly from the authoritative registry. The selected container is
// invocation context; no per-container mapping projection is retained.
const spec::Mapping* FindImportMapping(std::string_view container, std::string_view architecture,
                                       std::string_view component = {});
std::expected<void, Error> ValidateImportMapping(const spec::Mapping& mapping,
                                                 std::string_view container);

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
// A selected key stated as JSON null reads as absent (source_mapping.cc).
bool NullMeansAbsent(const spec::Mapping& m, std::string_view key, bool allow_unprefixed = false);
bool Pattern(std::string_view pattern);
std::optional<NameMatch> Match(std::string_view pattern, std::string_view name,
                               std::uint32_t layers);
}  // namespace normalization_detail

}  // namespace llmp::import

#endif  // LLMP_IMPORT_SOURCE_MAPPING_H_
