// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A test fixture's tensor table as a mapping's source (spec/mapping.h
// Source), for tests that synthesize a spec without an artifact: each
// tensor described by its GGML type or plain container dtype and shape in
// GGML order (`ne`), and the tensors that hold one tensor's parts (an MXFP8
// matrix's codes and scales, the n-gram table and its tensor scale,
// scale-atom expert tables) paired into it by an assembly table, as the
// native importer pairs them.

#ifndef LLMP_TESTS_SUPPORT_FIXTURE_SOURCE_H_
#define LLMP_TESTS_SUPPORT_FIXTURE_SOURCE_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "format/encoding.h"
#include "format/layout.h"
#include "format/tensor_format.h"
#include "spec/mapping.h"

namespace llmp::test_support {

// Tensors that hold one tensor's parts: a tensor `stem + suffixes[i]`
// holds part roles[i], where the stem matches `stem` (a pattern with
// {layer}; empty: any stem); a set is paired only where every part's
// tensor exists (otherwise they stay as they are).
struct FixtureAssembly {
  std::string_view stem{};
  std::array<std::string_view, 4> suffixes{};  // the parts' tensors; an empty one: none
  std::array<format::PartRole, 4> roles{};
  format::Encoding encoding = format::Encoding::kF32;
  format::Layout layout = format::Layout::kDense;
  std::uint8_t stacked = 1;
  bool experts = false;  // expert arrays
};

// Qwen3.8's ModelOpt fixture form: MXFP8 matrices, the n-gram table and
// its tensor scale, and the scale-atom expert tables.
inline constexpr std::array kQwen38ModelOptAssembly = {
    FixtureAssembly{.suffixes = {".weight", ".weight_scale", ""},
                    .roles = {format::PartRole::kCodes, format::PartRole::kBlockScales,
                              format::PartRole::kData},
                    .encoding = format::Encoding::kMxfp8,
                    .layout = format::Layout::kSplit},
    FixtureAssembly{
        .stem = "per_layer_token_embd",
        .suffixes = {".weight", ".weight_scale_2", ""},
        .roles = {format::PartRole::kData, format::PartRole::kTensorScale, format::PartRole::kData},
        .encoding = format::Encoding::kNvfp4,
        .layout = format::Layout::kRowInterleaved},
    FixtureAssembly{.stem = "blk.{layer}.ffn_gate_up_exps",
                    .suffixes = {".codes", ".scales", ""},
                    .roles = {format::PartRole::kCodes, format::PartRole::kBlockScales,
                              format::PartRole::kData},
                    .encoding = format::Encoding::kNvfp4,
                    .layout = format::Layout::kScaleAtoms,
                    .stacked = 2,
                    .experts = true},
    FixtureAssembly{.stem = "blk.{layer}.ffn_down_exps",
                    .suffixes = {".codes", ".scales", ""},
                    .roles = {format::PartRole::kCodes, format::PartRole::kBlockScales,
                              format::PartRole::kData},
                    .encoding = format::Encoding::kNvfp4,
                    .layout = format::Layout::kScaleAtoms,
                    .experts = true},
};

// A fixture tensor: its name, the roles it binds as (its name and tied
// aliases), its plain dtype or GGML type and GGML-ordered `ne`, and an
// expert array's expert count (0: none).
struct FixtureTensor {
  std::string name{};
  std::vector<std::string> roles{};
  bool plain = false;
  std::string type{};
  std::vector<std::uint64_t> ne{};
  std::uint32_t count = 0;
};

namespace fixture_detail {
inline bool Digit(char c) { return c >= '0' && c <= '9'; }
// Whether `name` is `pattern` with its {layer} a layer number.
inline bool Matches(std::string_view pattern, std::string_view name) {
  const std::size_t at = pattern.find(spec::kLayerField);
  if (at == std::string_view::npos) return pattern == name;
  const std::string_view before = pattern.substr(0, at);
  const std::string_view after = pattern.substr(at + spec::kLayerField.size());
  if (name.size() <= before.size() + after.size() || !name.starts_with(before) ||
      !name.ends_with(after))
    return false;
  const std::string_view digits =
      name.substr(before.size(), name.size() - before.size() - after.size());
  return digits.size() <= 6 && (digits.size() == 1 || digits.front() != '0') &&
         std::ranges::all_of(digits, Digit);
}
inline std::unexpected<std::string> Refuse(std::string_view what, std::string_view why) {
  return std::unexpected(std::format("{}: {}", what, why));
}

inline std::expected<format::TensorFormat, format::FormatError> TensorFormatOf(
    bool plain, std::string_view type, std::span<const std::uint64_t> ne) {
  const auto encoding = plain ? format::EncodingOfDtype(type) : format::EncodingOfGgmlType(type);
  const auto shape = format::ShapeOf(ne);
  if (!encoding || !shape) {
    return std::unexpected(!encoding ? format::FormatError::kUnknown : format::FormatError::kShape);
  }
  format::TensorFormat f{.encoding = *encoding, .layout = format::Layout::kDense, .shape = *shape};
  if (!plain) {
    const format::EncodingInfo& info = format::Info(*encoding);
    if (info.quantized) {
      f.layout = *encoding == format::Encoding::kNvfp4 ? format::Layout::kGgmlNvfp4Blocks
                                                       : format::Layout::kGgmlBlocks;
    }
  }
  if (auto ok = format::Validate(f); !ok) {
    return std::unexpected(ok.error());
  }
  return f;
}

struct PartInput {
  format::PartRole role = format::PartRole::kData;
  const FixtureTensor* tensor = nullptr;
};

inline std::expected<format::TensorFormat, format::FormatError> AssembleParts(
    format::Encoding encoding, format::Layout layout, std::span<const PartInput> parts,
    std::uint8_t stacked = 1) {
  if (static_cast<std::size_t>(encoding) >= format::Encodings().size() ||
      static_cast<std::size_t>(layout) >= format::Layouts().size()) {
    return std::unexpected(format::FormatError::kUnknown);
  }
  const auto find_part = [&](format::PartRole role) -> const FixtureTensor* {
    const auto it = std::ranges::find_if(parts, [&](const PartInput& p) { return p.role == role; });
    return it == parts.end() ? nullptr : it->tensor;
  };
  for (std::size_t i = 0; i < parts.size(); ++i) {
    if (parts[i].tensor == nullptr) return std::unexpected(format::FormatError::kParts);
    for (std::size_t j = i + 1; j < parts.size(); ++j) {
      if (parts[i].role == parts[j].role) return std::unexpected(format::FormatError::kParts);
    }
  }
  const format::EncodingInfo& info = format::Info(encoding);
  format::TensorFormat f{.encoding = encoding, .layout = layout, .stacked = stacked};
  f.params.tensor_scale = find_part(format::PartRole::kTensorScale) != nullptr;
  switch (layout) {
    case format::Layout::kSplit:
    case format::Layout::kScaleAtoms:
    case format::Layout::kRowInterleaved: {
      if (info.element_bits == 0 || !info.quantized) {
        return std::unexpected(format::FormatError::kLayout);
      }
      const format::PartRole role = layout == format::Layout::kRowInterleaved
                                        ? format::PartRole::kData
                                        : format::PartRole::kCodes;
      const FixtureTensor* rep = find_part(role);
      if (rep == nullptr) return std::unexpected(format::FormatError::kParts);
      const auto own = TensorFormatOf(rep->plain, rep->type, rep->ne);
      if (!own || own->layout != format::Layout::kDense ||
          format::Info(own->encoding).element_bits != 8 || own->shape.rank != 2) {
        return std::unexpected(format::FormatError::kParts);
      }
      const std::uint64_t row_bytes = own->shape.ne[0];
      const std::uint64_t rows = own->shape.ne[1];
      const std::uint64_t per_block = (info.block_elements * info.element_bits / 8) +
                                      (layout == format::Layout::kRowInterleaved ? 1 : 0);
      if (row_bytes % per_block != 0) {
        return std::unexpected(format::FormatError::kBlock);
      }
      f.shape = {.ne = {row_bytes / per_block * info.block_elements, rows, 1, 1}, .rank = 2};
      break;
    }
    case format::Layout::kDense:
    case format::Layout::kGgmlBlocks:
    case format::Layout::kGgmlNvfp4Blocks: {
      const FixtureTensor* rep = find_part(format::PartRole::kData);
      if (rep == nullptr) return std::unexpected(format::FormatError::kParts);
      auto own = TensorFormatOf(rep->plain, rep->type, rep->ne);
      if (!own) return own;
      if (own->encoding != encoding || own->layout != layout) {
        return std::unexpected(format::FormatError::kLayout);
      }
      f.shape = own->shape;
      break;
    }
    case format::Layout::kExl3Tiles:
      return std::unexpected(format::FormatError::kParts);
  }
  const auto extents = format::PartsOf(f);
  if (!extents) return std::unexpected(extents.error());
  if (extents->size() != parts.size()) return std::unexpected(format::FormatError::kParts);
  for (const format::PartExtent& extent : extents->parts()) {
    const FixtureTensor* rep = find_part(extent.role);
    if (rep == nullptr) return std::unexpected(format::FormatError::kParts);
    const auto own = TensorFormatOf(rep->plain, rep->type, rep->ne);
    if (!own) return std::unexpected(format::FormatError::kParts);
    const auto own_extents = format::PartsOf(*own);
    if (!own_extents || own_extents->size() != 1 ||
        own_extents->parts().front().bytes != extent.bytes) {
      return std::unexpected(format::FormatError::kParts);
    }
    if (extent.role == format::PartRole::kTensorScale &&
        (!rep->plain || format::EncodingOfDtype(rep->type) != format::Encoding::kF32)) {
      return std::unexpected(format::FormatError::kParts);
    }
  }
  return f;
}
}  // namespace fixture_detail

// The tensors as a source of `architecture` in `container`, with the folds
// its importer would record (a native import's; a GGUF's are its
// container's), the pairs `assembly` names assembled.
inline std::expected<spec::Source, std::string> FixtureSource(
    std::string architecture, std::string container, std::span<const FixtureTensor> tensors,
    std::span<const FixtureAssembly> assembly = {}, std::vector<std::string> folds = {}) {
  using fixture_detail::Refuse;
  spec::Source s{.architecture = std::move(architecture),
                 .container = std::move(container),
                 .tensors = {},
                 .folds = std::move(folds)};
  std::map<std::string, std::pair<format::TensorFormat, format::PartRole>, std::less<>> paired;
  std::map<std::string_view, const FixtureTensor*> by_name;
  for (const FixtureTensor& t : tensors) by_name.emplace(t.name, &t);
  for (const FixtureAssembly& a : assembly) {
    for (const FixtureTensor& t : tensors) {
      if ((t.count != 0) != a.experts || !t.name.ends_with(a.suffixes[0])) continue;
      const std::string_view stem =
          std::string_view(t.name).substr(0, t.name.size() - a.suffixes[0].size());
      if (!a.stem.empty() && !fixture_detail::Matches(a.stem, stem)) continue;
      std::vector<fixture_detail::PartInput> parts;
      std::vector<std::string> part_names;
      std::size_t wanted = 0;
      for (std::size_t i = 0; i < a.suffixes.size(); ++i) {
        if (a.suffixes.at(i).empty()) continue;
        ++wanted;
        std::string name = std::format("{}{}", stem, a.suffixes.at(i));
        const auto found = by_name.find(name);
        if (found == by_name.end() || (found->second->count != 0) != a.experts) break;
        parts.push_back({.role = a.roles.at(i), .tensor = found->second});
        part_names.push_back(std::move(name));
      }
      if (parts.size() != wanted) continue;  // not every part: left as it is
      auto f = fixture_detail::AssembleParts(a.encoding, a.layout, parts, a.stacked);
      if (!f)
        return Refuse(t.name, std::format("its parts do not assemble ({})",
                                          format::FormatErrorName(f.error())));
      if (a.experts) {
        f->shape.ne[2] = t.count;
        f->shape.rank = 3;
      }
      for (std::size_t i = 0; i < parts.size(); ++i)
        if (!paired.emplace(part_names[i], std::pair{*f, parts[i].role}).second)
          return Refuse(part_names[i], "a part of two tensors");
    }
  }
  for (const FixtureTensor& t : tensors) {
    auto f = fixture_detail::TensorFormatOf(t.plain, t.type, t.ne);
    if (!f || (t.count != 0 && f->shape.rank > 2))
      return Refuse(t.name,
                    f ? std::string("no two-dimensional slice format")
                      : std::format("no tensor format ({})", format::FormatErrorName(f.error())));
    format::PartRole part = format::PartRole::kData;
    if (const auto at = paired.find(t.name); at != paired.end()) {
      f = at->second.first;
      part = at->second.second;
    } else if (t.count != 0) {
      f->shape.ne[2] = t.count;
      f->shape.rank = 3;
    }
    if (t.count != 0) {
      s.tensors.push_back({.name = t.name, .format = *f, .part = part, .experts = true});
      continue;
    }
    for (const std::string& role : t.roles)
      s.tensors.push_back({.name = role, .format = *f, .part = part, .experts = false});
  }
  return s;
}

}  // namespace llmp::test_support

#endif  // LLMP_TESTS_SUPPORT_FIXTURE_SOURCE_H_
