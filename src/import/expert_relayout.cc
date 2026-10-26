// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "import/expert_relayout.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <format>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <system_error>
#include <utility>

#include "components/component.h"
#include "format/encoding.h"
#include "format/layout.h"
#include "spec/mapping.h"
#include "spec/spec.h"

namespace llmp::import {
namespace {

using format::Encoding;
using format::Layout;
using format::PartRole;

// The encodings whose routed experts are relaid out: those whose grouped
// products read the scale atoms today (NVFP4's W4A4 grouped GEMM and the
// format ladder's atom decoders).
constexpr std::array kEncodings = {Encoding::kNvfp4, Encoding::kMxfp8, Encoding::kMxfp4};
// The encodings a row-gathered table is relaid out to row_interleaved for:
// those a device kernel reads in that layout (NVFP4's llmp.nvfp4.get_rows
// and the format ladder's nvfp4.row_interleaved decoder). Another joins
// with its readers; until then its table stays split, which its row
// lookups read.
constexpr std::array kRowTableEncodings = {Encoding::kNvfp4};

std::unexpected<Error> Bad(std::string_view why) {
  return std::unexpected(Error{.rule = Rule::kConfig, .reason = why});
}

const PartAssembly* PartOf(const TensorAssembly& a, PartRole role) {
  const auto found = std::ranges::find(a.parts, role, &PartAssembly::part);
  return found == a.parts.end() ? nullptr : &*found;
}

// The relaid-out table of `gate` (with `up` stacked after it, expert by
// expert, when given).
std::expected<TensorAssembly, Error> Relaid(const TensorAssembly& gate, const TensorAssembly* up,
                                            std::string name) {
  TensorAssembly out{.name = std::move(name), .format = gate.format};
  out.format.layout = Layout::kScaleAtoms;
  out.format.params.tensor_scale = false;
  out.relayout_from = Layout::kSplit;
  if (up != nullptr) {
    if (out.format.shape.ne[1] > std::numeric_limits<std::uint64_t>::max() / 2)
      return Bad("stacked expert table height overflows");
    out.format.stacked = 2;
    out.format.shape.ne[1] *= 2;
  }
  if (!format::PartsOf(out.format)) return Bad("relaid expert table format has invalid parts");
  for (const PartRole role : {PartRole::kCodes, PartRole::kBlockScales}) {
    const PartAssembly* g = PartOf(gate, role);
    PartAssembly part{.part = role, .kind = PartAssemblyKind::kMatrices, .sources = {}};
    if (up == nullptr) {
      part.sources = g->sources;
    } else {
      const PartAssembly* u = PartOf(*up, role);
      part.sources.reserve(2 * g->sources.size());
      for (std::size_t e = 0; e < g->sources.size(); ++e) {
        part.sources.push_back(g->sources[e]);
        part.sources.push_back(u->sources[e]);
      }
    }
    out.parts.push_back(std::move(part));
  }
  return out;
}

std::string ExpandPattern(std::string_view pattern, std::uint32_t layer) {
  const auto at = pattern.find(spec::kLayerField);
  if (at == std::string_view::npos) return std::string(pattern);
  return std::format("{}{}{}", pattern.substr(0, at), layer,
                     pattern.substr(at + spec::kLayerField.size()));
}

// A mapping's tensor rule's role's import transform (the catalog's, by its
// slot's component), or null.
const components::ImportTransform* TransformOf(const spec::Mapping& m, const spec::TensorRule& r) {
  if (r.role.empty()) return nullptr;
  const auto slot = std::ranges::find(m.slots, r.slot, &spec::SlotRule::slot);
  return slot == m.slots.end() ? nullptr
                               : components::ImportTransformOf(slot->kind, slot->version, r.role);
}

// The stacked table's name for the gate rule `gate` at `layer`: the rule of
// the role its transform stacks into, in the same slot, holding codes or
// data (a ".weight" spelling first).
std::string StackedNameFromRules(std::span<const spec::TensorRule> tensors,
                                 const spec::TensorRule& gate, std::string_view stacked,
                                 std::uint32_t layer, bool require_weight_suffix) {
  for (const auto& s : tensors) {
    if (s.slot != gate.slot || s.role != stacked ||
        (s.part != PartRole::kCodes && s.part != PartRole::kData))
      continue;
    if (require_weight_suffix && !s.pattern.ends_with(".weight")) continue;
    return ExpandPattern(s.pattern, layer);
  }
  return {};
}

std::string StackedName(std::span<const spec::TensorRule> tensors, const spec::TensorRule& gate,
                        std::string_view stacked, std::uint32_t layer) {
  if (stacked.empty()) return {};
  for (const bool require_weight : {true, false})
    if (auto s = StackedNameFromRules(tensors, gate, stacked, layer, require_weight); !s.empty())
      return s;
  return {};
}

std::optional<std::uint32_t> MatchLayer(std::string_view pattern, std::string_view name) {
  const auto at = pattern.find(spec::kLayerField);
  if (at == std::string_view::npos) {
    return pattern == name ? std::optional<std::uint32_t>(0) : std::nullopt;
  }
  const auto before = pattern.substr(0, at);
  const auto after = pattern.substr(at + spec::kLayerField.size());
  if (name.size() <= before.size() + after.size() || !name.starts_with(before) ||
      !name.ends_with(after)) {
    return std::nullopt;
  }
  const auto digits = name.substr(before.size(), name.size() - before.size() - after.size());
  if (digits.empty() || digits.size() > 5 || (digits.size() > 1 && digits[0] == '0')) {
    return std::nullopt;
  }
  std::uint32_t layer = 0;
  const auto parsed = std::from_chars(digits.data(), digits.data() + digits.size(), layer);
  if (parsed.ec != std::errc{} || parsed.ptr != digits.data() + digits.size() ||
      layer >= (1U << 16U)) {
    return std::nullopt;
  }
  return layer;
}

// A routed-expert table of `m`'s rules: the role its rule binds declares
// an expert projection (components::ImportTransform). Its MoE is its
// layer's, a drafter's apart from its target's.
std::optional<ExpertTable> MatchInRules(const spec::Mapping& m,
                                        std::span<const spec::TensorRule> tensors,
                                        std::string_view name, std::uint32_t output) {
  for (const auto& r : tensors) {
    if (r.part != PartRole::kData && r.part != PartRole::kCodes) continue;
    const components::ImportTransform* transform = TransformOf(m, r);
    if (transform == nullptr || transform->expert == components::ExpertProjection::kNone) continue;
    const auto layer = MatchLayer(r.pattern, name);
    if (!layer) continue;
    const auto proj =
        transform->expert == components::ExpertProjection::kGate ? ExpertProjection::kGate
        : transform->expert == components::ExpertProjection::kUp ? ExpertProjection::kUp
                                                                 : ExpertProjection::kDown;
    const std::uint32_t moe_base = m.role == spec::StageRole::kDrafter ? (1U << 16U) : 0U;
    return ExpertTable{.output = output,
                       .projection = proj,
                       .moe = moe_base + *layer,
                       .name = std::string(name),
                       .stacked = proj == ExpertProjection::kGate
                                      ? StackedName(tensors, r, transform->stacked, *layer)
                                      : "",
                       .scales = std::string(name) + "_scale_2"};
  }
  return std::nullopt;
}

std::optional<ExpertTable> MatchInMapping(const spec::Mapping& m, std::string_view name,
                                          std::uint32_t output) {
  if (auto t = MatchInRules(m, m.tensors, name, output)) return t;
  for (const auto& d : m.dialects) {
    if (auto t = MatchInRules(m, d.tensors, name, output)) return t;
  }
  return std::nullopt;
}

}  // namespace

std::expected<std::optional<ExpertTable>, Error> ExpertTableOf(const TensorAssembly& tensor,
                                                               std::uint32_t output,
                                                               const spec::Mapping& mapping) {
  if (auto t = MatchInMapping(mapping, tensor.name, output)) return t;
  if (mapping.companion != nullptr) {
    if (auto t = MatchInMapping(*mapping.companion, tensor.name, output)) return t;
  }
  return std::optional<ExpertTable>{};
}

bool RelaysOut(const format::TensorFormat& t) {
  if (t.layout != Layout::kSplit || t.shape.rank != 3 || t.stacked != 1 ||
      !std::ranges::contains(kEncodings, t.encoding)) {
    return false;
  }
  const auto block = format::Info(t.encoding).block_elements;
  if (block == 0 || t.shape.ne[0] % block != 0) return false;
  const auto blocks = t.shape.ne[0] / block;
  return t.shape.ne[1] != 0 && t.shape.ne[1] % format::kScaleAtomRows == 0 &&
         blocks % format::kScaleAtomBlocks == 0;
}

std::expected<ExpertRelayoutSummary, Error> RelayoutExpertTables(
    std::vector<TensorAssembly>& outputs, std::span<const ExpertTable> tables) {
  std::vector<const ExpertTable*> listed(outputs.size(), nullptr);
  for (const ExpertTable& t : tables) {
    if (t.output >= outputs.size() || listed[t.output] != nullptr || t.name.empty())
      return Bad("an expert table outside the outputs, listed twice or unnamed");
    const TensorAssembly& a = outputs[t.output];
    listed[t.output] = &t;
    // A table of another encoding or layout is kept as it is assembled (an
    // MLX drafter's pre-stacked affine experts); only one relaid out must be
    // gathered expert by expert.
    if (!RelaysOut(a.format)) continue;
    if (a.permutation || a.relayout_from || a.format.shape.rank != 3 ||
        PartOf(a, PartRole::kCodes) == nullptr || PartOf(a, PartRole::kBlockScales) == nullptr ||
        std::ranges::any_of(a.parts, [&](const PartAssembly& p) {
          return p.kind != PartAssemblyKind::kMatrices || p.sources.size() != a.format.shape.ne[2];
        }))
      return Bad("an expert table that is not a whole gathered table");
  }
  // Each relaid gate's up, by output.
  std::vector<std::optional<std::uint32_t>> partner(outputs.size());
  std::vector<bool> absorbed(outputs.size(), false);
  for (const ExpertTable& g : tables) {
    if (g.projection != ExpertProjection::kGate || !RelaysOut(outputs[g.output].format)) continue;
    for (const ExpertTable& u : tables) {
      if (u.projection == ExpertProjection::kUp && u.moe == g.moe &&
          outputs[u.output].format == outputs[g.output].format && !absorbed[u.output]) {
        partner[g.output] = u.output;
        absorbed[u.output] = true;
        break;
      }
    }
  }
  ExpertRelayoutSummary summary;
  std::vector<TensorAssembly> rewritten, scales;
  rewritten.reserve(outputs.size());
  const auto add_scales = [&](const TensorAssembly& a, const ExpertTable& t) {
    const PartAssembly* part = PartOf(a, PartRole::kTensorScale);
    if (part == nullptr) return !a.format.params.tensor_scale;
    if (t.scales.empty() || part->sources.size() != a.format.shape.ne[2]) return false;
    TensorAssembly v{.name = t.scales,
                     .format = {.encoding = Encoding::kF32,
                                .layout = Layout::kDense,
                                .shape = {.ne = {part->sources.size(), 1, 1, 1}, .rank = 1}}};
    v.parts.push_back(
        {.part = PartRole::kData, .kind = PartAssemblyKind::kScalars, .sources = part->sources});
    scales.push_back(std::move(v));
    ++summary.scale_vectors;
    return true;
  };
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const ExpertTable* t = listed[i];
    if (t == nullptr || !RelaysOut(outputs[i].format)) {
      auto& out = outputs[i];
      if (t != nullptr && out.format.encoding == Encoding::kNvfp4 &&
          out.format.layout == Layout::kSplit && out.format.shape.rank == 3 &&
          out.format.params.tensor_scale) {
        if (!add_scales(out, *t)) return Bad("an expert table's tensor scales have no vector name");
        std::erase_if(out.parts,
                      [](const auto& part) { return part.part == PartRole::kTensorScale; });
        out.format.params.tensor_scale = false;
        out.separate_tensor_scale = true;
      }
      rewritten.push_back(std::move(out));
      continue;
    }
    if (absorbed[i]) continue;  // stacked with its gate
    const TensorAssembly* up = partner[i] ? &outputs[*partner[i]] : nullptr;
    if (!add_scales(outputs[i], *t) || (up != nullptr && !add_scales(*up, *listed[*partner[i]])))
      return Bad("an expert table's tensor scales have no vector name");
    if (up != nullptr && t->stacked.empty()) return Bad("a stacked gate and up table has no name");
    auto relaid = Relaid(outputs[i], up, up != nullptr ? t->stacked : t->name);
    if (!relaid) return std::unexpected(relaid.error());
    rewritten.push_back(std::move(*relaid));
    ++summary.tables;
    if (up != nullptr) ++summary.stacked;
  }
  for (auto& v : scales) rewritten.push_back(std::move(v));
  outputs = std::move(rewritten);
  return summary;
}

std::expected<ExpertRelayoutSummary, Error> RelayoutExpertTables(
    std::vector<TensorAssembly>& outputs, const spec::Mapping& mapping) {
  std::vector<ExpertTable> tables;
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    auto table = ExpertTableOf(outputs[i], static_cast<std::uint32_t>(i), mapping);
    if (!table) return std::unexpected(table.error());
    if (*table) tables.push_back(std::move(**table));
  }
  return RelayoutExpertTables(outputs, tables);
}

std::expected<bool, Error> RelayoutRowTable(TensorAssembly& table) {
  const auto& f = table.format;
  const PartAssembly* codes = PartOf(table, PartRole::kCodes);
  const PartAssembly* scales = PartOf(table, PartRole::kBlockScales);
  const PartAssembly* tensor_scale = PartOf(table, PartRole::kTensorScale);
  if (f.layout != Layout::kSplit || f.shape.rank != 2 || f.stacked != 1 ||
      !std::ranges::contains(kRowTableEncodings, f.encoding) || table.permutation ||
      table.relayout_from || codes == nullptr || scales == nullptr ||
      codes->kind != PartAssemblyKind::kRows || scales->kind != PartAssemblyKind::kRows ||
      codes->sources.size() != scales->sources.size() ||
      table.parts.size() != (tensor_scale != nullptr ? 3U : 2U) ||
      (tensor_scale != nullptr && tensor_scale->kind != PartAssemblyKind::kSharedTensorScale))
    return false;
  PartAssembly data{.part = PartRole::kData, .kind = PartAssemblyKind::kInterleavedRows};
  data.sources.reserve(2 * codes->sources.size());
  for (std::size_t i = 0; i < codes->sources.size(); ++i) {
    data.sources.push_back(codes->sources[i]);
    data.sources.push_back(scales->sources[i]);
  }
  std::vector<PartAssembly> parts;
  parts.push_back(std::move(data));
  if (tensor_scale != nullptr) parts.push_back(*tensor_scale);
  table.parts = std::move(parts);
  table.format.layout = Layout::kRowInterleaved;
  table.relayout_from = Layout::kSplit;
  return true;
}

}  // namespace llmp::import
