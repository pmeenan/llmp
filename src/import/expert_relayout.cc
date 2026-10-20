// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "import/expert_relayout.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <format>
#include <optional>
#include <string_view>
#include <system_error>
#include <utility>

#include "format/encoding.h"
#include "format/layout.h"
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
constexpr std::uint64_t kMaxBlocks = 256;  // PartRoutePlan::ScaleAtoms

std::unexpected<Error> Bad(std::string_view why) {
  return std::unexpected(Error{.rule = Rule::kConfig, .reason = why});
}

const PartAssembly* PartOf(const TensorAssembly& a, PartRole role) {
  const auto found = std::ranges::find(a.parts, role, &PartAssembly::part);
  return found == a.parts.end() ? nullptr : &*found;
}

// The relaid-out table of `gate` (with `up` stacked after it, expert by
// expert, when given).
TensorAssembly Relaid(const TensorAssembly& gate, const TensorAssembly* up, std::string name) {
  TensorAssembly out{.name = std::move(name), .format = gate.format};
  out.format.layout = Layout::kScaleAtoms;
  out.format.params.tensor_scale = false;
  out.relayout_from = Layout::kSplit;
  if (up != nullptr) {
    out.format.stacked = 2;
    out.format.shape.ne[1] *= 2;
  }
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

std::string StackedNameOf(std::string_view name) {
  if (const auto at = name.find("ffn_gate"); at != std::string_view::npos) {
    std::string s(name);
    s.replace(at, 8, "ffn_gate_up");
    return s;
  }
  if (const auto at = name.find("gate"); at != std::string_view::npos) {
    std::string s(name);
    s.replace(at, 4, "gate_up");
    return s;
  }
  return {};
}

}  // namespace

std::optional<ExpertTable> ExpertTableOf(std::string_view name, std::uint32_t output) {
  constexpr std::string_view kSuffix = "_exps.weight";
  if (!name.ends_with(kSuffix)) return std::nullopt;
  std::string_view prefix = "";
  std::uint32_t moe_base = 0;
  if (name.starts_with("blk.")) {
    prefix = "blk.";
  } else if (name.starts_with("mtp.blk.")) {
    prefix = "mtp.blk.";
    moe_base = 10000;
  } else if (name.starts_with("mtp.")) {
    prefix = "mtp.";
    moe_base = 10000;
  } else {
    return std::nullopt;
  }
  auto rest = name.substr(prefix.size(), name.size() - prefix.size() - kSuffix.size());
  const auto dot = rest.find('.');
  if (dot == std::string_view::npos || dot == 0 || dot > 5 || (dot > 1 && rest[0] == '0'))
    return std::nullopt;
  std::uint32_t layer = 0;
  const auto parsed = std::from_chars(rest.data(), rest.data() + dot, layer);
  if (parsed.ec != std::errc{} || parsed.ptr != rest.data() + dot || layer >= (1U << 16U))
    return std::nullopt;
  const auto projection = rest.substr(dot + 1);
  ExpertTable t{.output = output, .moe = moe_base + layer, .name = std::string(name)};
  if (projection == "ffn_gate") {
    t.projection = ExpertProjection::kGate;
    t.stacked = std::format("{}{}.ffn_gate_up_exps.weight", prefix, layer);
  } else if (projection == "ffn_up") {
    t.projection = ExpertProjection::kUp;
  } else if (projection == "ffn_down") {
    t.projection = ExpertProjection::kDown;
  } else {
    return std::nullopt;
  }
  t.scales = std::string(name) + "_scale_2";
  return t;
}

std::expected<std::optional<ExpertTable>, Error> ExpertTableOf(const TensorAssembly& tensor,
                                                               std::uint32_t output,
                                                               const spec::ModelSpec* spec) {
  if (spec != nullptr) {
    std::uint32_t moe_id = 0;
    for (const auto& stage : spec->stages) {
      for (std::size_t l = 0; l < stage.layers.size(); ++l) {
        for (const auto& c : stage.layers[l].components) {
          if (c.kind != "ffn.moe") continue;
          const std::uint32_t current_moe = moe_id++;
          for (const auto& w : c.weights) {
            if (w.tensor != tensor.name) continue;
            if (!w.stage.empty() ||
                (w.part != format::PartRole::kData && w.part != format::PartRole::kCodes))
              continue;
            ExpertProjection proj = ExpertProjection::kOther;
            if (w.role == "gate_experts" || w.role == "routed_gate")
              proj = ExpertProjection::kGate;
            else if (w.role == "up_experts" || w.role == "routed_up")
              proj = ExpertProjection::kUp;
            else if (w.role == "down_experts" || w.role == "routed_down")
              proj = ExpertProjection::kDown;
            if (proj == ExpertProjection::kOther) continue;
            ExpertTable t{
                .output = output,
                .projection = proj,
                .moe = current_moe,
                .name = tensor.name,
                .stacked = proj == ExpertProjection::kGate ? StackedNameOf(tensor.name) : "",
                .scales = tensor.name + "_scale_2"};
            return t;
          }
        }
      }
      for (std::size_t m = 0; m < stage.heads.mtp.size(); ++m) {
        const auto& c = stage.heads.mtp[m];
        if (c.kind != "ffn.moe") continue;
        const std::uint32_t current_moe = moe_id++;
        for (const auto& w : c.weights) {
          if (w.tensor != tensor.name) continue;
          if (!w.stage.empty() ||
              (w.part != format::PartRole::kData && w.part != format::PartRole::kCodes))
            continue;
          ExpertProjection proj = ExpertProjection::kOther;
          if (w.role == "gate_experts" || w.role == "routed_gate")
            proj = ExpertProjection::kGate;
          else if (w.role == "up_experts" || w.role == "routed_up")
            proj = ExpertProjection::kUp;
          else if (w.role == "down_experts" || w.role == "routed_down")
            proj = ExpertProjection::kDown;
          if (proj == ExpertProjection::kOther) continue;
          ExpertTable t{
              .output = output,
              .projection = proj,
              .moe = current_moe,
              .name = tensor.name,
              .stacked = proj == ExpertProjection::kGate ? StackedNameOf(tensor.name) : "",
              .scales = tensor.name + "_scale_2"};
          return t;
        }
      }
    }
  }
  return ExpertTableOf(tensor.name, output);
}

std::expected<std::optional<ExpertTable>, Error> ExpertTableOf(const TensorAssembly& tensor,
                                                               const spec::ModelSpec* spec) {
  return ExpertTableOf(tensor, 0, spec);
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
         blocks % format::kScaleAtomBlocks == 0 && blocks <= kMaxBlocks;
}

std::expected<ExpertRelayoutSummary, Error> RelayoutExpertTables(
    std::vector<TensorAssembly>& outputs, std::span<const ExpertTable> tables) {
  std::vector<const ExpertTable*> listed(outputs.size(), nullptr);
  for (const ExpertTable& t : tables) {
    if (t.output >= outputs.size() || listed[t.output] != nullptr || t.name.empty())
      return Bad("an expert table outside the outputs, listed twice or unnamed");
    const TensorAssembly& a = outputs[t.output];
    if (a.permutation || a.relayout_from || a.format.shape.rank != 3 ||
        PartOf(a, PartRole::kCodes) == nullptr || PartOf(a, PartRole::kBlockScales) == nullptr ||
        std::ranges::any_of(a.parts, [&](const PartAssembly& p) {
          return p.kind != PartAssemblyKind::kMatrices || p.sources.size() != a.format.shape.ne[2];
        }))
      return Bad("an expert table that is not a whole gathered table");
    listed[t.output] = &t;
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
    if (part == nullptr) return true;
    if (t.scales.empty()) return false;
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
      rewritten.push_back(std::move(outputs[i]));
      continue;
    }
    if (absorbed[i]) continue;  // stacked with its gate
    const TensorAssembly* up = partner[i] ? &outputs[*partner[i]] : nullptr;
    if (!add_scales(outputs[i], *t) || (up != nullptr && !add_scales(*up, *listed[*partner[i]])))
      return Bad("an expert table's tensor scales have no vector name");
    if (up != nullptr && t->stacked.empty()) return Bad("a stacked gate and up table has no name");
    rewritten.push_back(Relaid(outputs[i], up, up != nullptr ? t->stacked : t->name));
    ++summary.tables;
    if (up != nullptr) ++summary.stacked;
  }
  for (auto& v : scales) rewritten.push_back(std::move(v));
  outputs = std::move(rewritten);
  return summary;
}

std::expected<ExpertRelayoutSummary, Error> RelayoutExpertTables(
    std::vector<TensorAssembly>& outputs, const spec::ModelSpec* spec) {
  std::vector<ExpertTable> tables;
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    auto table = ExpertTableOf(outputs[i], static_cast<std::uint32_t>(i), spec);
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
