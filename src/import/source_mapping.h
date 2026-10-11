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
  // The only additional operation needed by these source domains.
  std::string_view key{};
  std::int64_t reject_equal = 0;
};
struct AuxiliaryNamespace {
  std::string_view prefix{};  // literal, dot-terminated; metadata only
  std::string_view reason{};
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
};

const SourceMapping* FindSourceMapping(std::string_view container, std::string_view architecture);
// Allocation-free validation of the bounded normalization schema, including
// aliases' whole-pattern overlap. Full spec::ValidateMapping runs after funding.
std::expected<void, Error> ValidateSourceMapping(const SourceMapping& mapping);

namespace normalization_detail {
inline constexpr std::uint32_t kLayerLimit = 1024;
inline constexpr std::size_t kRuleLimit = 32;
// Bits for the accepted source scalar and layer-list alternatives. Int and Real intersect
// when one key supplies both; a bool is never an integer.
inline constexpr std::uint8_t kInteger = 1, kReal = 2, kFlag = 4, kText = 8, kIntegers = 16;
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
