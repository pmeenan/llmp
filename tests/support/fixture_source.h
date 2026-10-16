// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A test fixture's tensor table as a mapping's source (spec/mapping.h
// Source), for tests that synthesize a spec without an artifact: each
// tensor described as a v0 descriptor gives it (a GGML type and ne, a plain
// dtype and shape, an EXL3 part; artifact/format_map.h FormatOf), and the
// tensors that hold one tensor's parts (an MXFP8 matrix's codes and scales,
// an EXL3 trellis's four parts, scale-atom expert tables) paired into it by
// an assembly table, as the native importer pairs them.
//
// Also the same of a v0 artifact (its resources and expert arrays), which
// the Qwen-Image component tests still read: the image slice moves that
// pipeline to stored spec data (D-118) and retires that use. Serving reads
// v1 artifacts only (spec/stored.h SourceOf).

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

#include "artifact/artifact.h"
#include "artifact/format_map.h"
#include "artifact/representation.h"
#include "format/encoding.h"
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

// An EXL3 trellis's four parts (the approved fixtures use the mcg marker).
inline constexpr std::array kExl3Assembly = {
    FixtureAssembly{.suffixes = {".trellis", ".suh", ".svh", ".mcg"},
                    .roles = {format::PartRole::kCodes, format::PartRole::kInputSigns,
                              format::PartRole::kOutputSigns, format::PartRole::kCodebookMarker},
                    .encoding = format::Encoding::kExl3Trellis,
                    .layout = format::Layout::kExl3Tiles},
};

// A fixture tensor: its name, the roles it binds as (its name and tied
// aliases), its descriptor, and an expert array's expert count (0: none).
struct FixtureTensor {
  std::string name{};
  std::vector<std::string> roles{};
  const artifact::Representation* repr = nullptr;
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
      std::vector<artifact::V0Part> parts;
      std::vector<std::string> part_names;
      std::size_t wanted = 0;
      for (std::size_t i = 0; i < a.suffixes.size(); ++i) {
        if (a.suffixes.at(i).empty()) continue;
        ++wanted;
        std::string name = std::format("{}{}", stem, a.suffixes.at(i));
        const auto found = by_name.find(name);
        if (found == by_name.end() || (found->second->count != 0) != a.experts ||
            found->second->repr == nullptr)
          break;
        parts.push_back({.role = a.roles.at(i), .rep = found->second->repr});
        part_names.push_back(std::move(name));
      }
      if (parts.size() != wanted) continue;  // not every part: left as it is
      auto f = artifact::Assemble(a.encoding, a.layout, parts, a.stacked);
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
    if (t.repr == nullptr) return Refuse(t.name, "no representation");
    auto f = artifact::FormatOf(*t.repr);
    if (!f || (t.count != 0 && f->shape.rank > 2))
      return Refuse(t.name,
                    f ? std::string("no two-dimensional slice format")
                      : std::format("no tensor format ({})", format::FormatErrorName(f.error())));
    format::PartRole part = artifact::V0PartRole(*t.repr);
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

// A v0 artifact's resources and expert arrays as a source (the container
// "gguf" when every source file is a GGUF, "safetensors" when every one is
// a safetensors or JSON file, else "other"). The Qwen-Image component
// tests' (the image slice's, until its components are stored v1 specs).
inline std::expected<spec::Source, std::string> V0ArtifactSource(
    const artifact::Artifact& artifact, std::span<const FixtureAssembly> assembly = {}) {
  if (artifact.format_version() != 0)
    return fixture_detail::Refuse(std::format("artifact {}", artifact.id()), "not format v0");
  const auto sources = artifact.sources();
  const auto all = [&](auto pred) {
    return !sources.empty() && std::ranges::all_of(sources, pred);
  };
  std::string container = "other";
  if (all([](const artifact::Source& f) { return f.name.ends_with(".gguf"); }))
    container = "gguf";
  else if (all([](const artifact::Source& f) {
             return f.name.ends_with(".safetensors") || f.name.ends_with(".json");
           }))
    container = "safetensors";
  std::vector<FixtureTensor> tensors;
  for (const artifact::Resource& r : artifact.resources())
    tensors.push_back({.name = r.name, .roles = r.roles, .repr = &r.repr, .count = 0});
  for (const artifact::ExpertArray& a : artifact.expert_arrays())
    tensors.push_back({.name = a.name, .roles = {a.name}, .repr = &a.repr, .count = a.count});
  return FixtureSource(artifact.model().architecture, std::move(container), tensors, assembly);
}

}  // namespace llmp::test_support

#endif  // LLMP_TESTS_SUPPORT_FIXTURE_SOURCE_H_
