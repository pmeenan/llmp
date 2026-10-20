// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "import/indexed_parts.h"

#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <numeric>
#include <tuple>
#include <utility>

#include "format/encoding.h"

namespace llmp::import {
namespace {
Error Bad(std::string_view why, Rule rule = Rule::kContainer) {
  return {.rule = rule, .reason = why};
}
bool Add(std::uint64_t& a, std::uint64_t b) {
  if (b > std::numeric_limits<std::uint64_t>::max() - a) return false;
  a += b;
  return true;
}
bool IsFoldSourceEncoding(format::Encoding e) {
  return e == format::Encoding::kF32 || e == format::Encoding::kF16 || e == format::Encoding::kBF16;
}
bool SameEncoding(const format::TensorFormat& a, const format::TensorFormat& b) {
  return a.encoding == b.encoding && a.params == b.params && a.layout == b.layout &&
         a.stacked == b.stacked;
}
bool DenseReshape(const format::TensorFormat& a, const format::TensorFormat& b) {
  if (!SameEncoding(a, b)) return false;
  const bool ok_dense =
      a.layout == format::Layout::kDense &&
      (a.encoding == format::Encoding::kBF16 || a.encoding == format::Encoding::kF32 ||
       a.encoding == format::Encoding::kF16);
  const bool ok_split =
      a.layout == format::Layout::kSplit &&
      (a.encoding == format::Encoding::kAffine || a.encoding == format::Encoding::kMxfp8 ||
       a.encoding == format::Encoding::kMxfp4 || a.encoding == format::Encoding::kNvfp4);
  if (!ok_dense && !ok_split) return false;
  // Only remove/insert singleton axes, retaining the order of actual axes.
  std::array<std::uint64_t, 4> left{}, right{};
  std::size_t x = 0, y = 0;
  for (auto n : a.shape.ne)
    if (n != 1) left[x++] = n;
  for (auto n : b.shape.ne)
    if (n != 1) right[y++] = n;
  return left == right;
}
bool GatherEncoding(const format::TensorFormat& f) {
  return f.layout == format::Layout::kDense ||
         (f.layout == format::Layout::kSplit &&
          (f.encoding == format::Encoding::kNvfp4 || f.encoding == format::Encoding::kMxfp8 ||
           f.encoding == format::Encoding::kMxfp4 || f.encoding == format::Encoding::kAffine));
}
}  // namespace

std::expected<IndexedPartCost, Error> IndexedPartPlan::Bound(
    const Checkpoint& source, std::span<const TensorAssembly> outputs) {
  if (source.tensors.empty() || source.tensors.size() > kMaxTensors || outputs.empty() ||
      outputs.size() > kMaxTensors || source.files.empty() || source.files.size() > 256 ||
      !source.components.empty())
    return std::unexpected(Bad("indexed assembly count outside bound", Rule::kBounds));
  for (const auto& f : source.files)
    if (f.name.size() > kMaxNameBytes || f.path.native().size() > 4096)
      return std::unexpected(Bad("indexed source file name outside bound", Rule::kBounds));
  IndexedPartCost cost{};
  for (const auto& t : source.tensors)
    if (t.name.empty() || t.name.size() > kMaxNameBytes || !NoControlCharacters(t.name))
      return std::unexpected(Bad("indexed source name outside bound", Rule::kBounds));
  for (const auto& out : outputs) {
    if (out.name.empty() || out.name.size() > kMaxNameBytes || !NoControlCharacters(out.name) ||
        out.parts.empty() || out.parts.size() > format::PartList::kMax)
      return std::unexpected(Bad("indexed output outside bound", Rule::kBounds));
    for (const auto& part : out.parts) {
      if (part.sources.empty() || part.sources.size() > source.tensors.size() ||
          !Add(cost.members, part.sources.size()) || cost.members > source.tensors.size())
        return std::unexpected(Bad("indexed members outside bound", Rule::kBounds));
    }
  }
  // Exact reserved vector capacities below: retained names/dispositions;
  // temporary source-extent order and used flags; outputs plus temporary name
  // order; member records and physical order. Charge each heap allocation 64
  // bytes and every output name at twice its maximum extent, conservatively
  // including SSO. std::sort uses bounded stack, included in the 64KiB floor.
  cost.metadata_bytes = 65536 + 8 * 64;
  cost.metadata_bytes +=
      source.tensors.size() * (2 * sizeof(std::uint32_t) + sizeof(SourceDisposition) + 1);
  cost.metadata_bytes += outputs.size() * (sizeof(IndexedOutput) + sizeof(std::uint32_t) +
                                           2 * (kMaxNameBytes + 1) + 64);
  cost.metadata_bytes += cost.members * (sizeof(IndexedMember) + sizeof(std::uint32_t));
  // Name comparisons inspect at most512 bytes. Both source sorts and output
  // sort are comparison-bounded by a conservative64*n*bit_width(n) allowance;
  // includes all format/member checks and the <=256-file identity checks.
  const auto n = static_cast<std::uint64_t>(source.tensors.size() + outputs.size());
  cost.work_units = 256ULL * 256 * 4096 +
                    64 * n * static_cast<std::uint64_t>(std::bit_width(n)) * (kMaxNameBytes + 1);
  return cost;
}

std::expected<IndexedPartPlan, Error> IndexedPartPlan::Make(
    Checkpoint&& source, std::span<const TensorAssembly> outputs,
    std::span<const SourceDisposition> dispositions, IndexedPartCost prepaid) {
  auto bound = Bound(source, outputs);
  if (!bound) return std::unexpected(bound.error());
  if (prepaid.metadata_bytes < bound->metadata_bytes || prepaid.work_units < bound->work_units ||
      prepaid.members < bound->members || dispositions.size() != source.tensors.size())
    return std::unexpected(
        Bad("indexed assembly lacks complete prepayment/dispositions", Rule::kBounds));
  for (std::size_t i = 0; i < source.files.size(); ++i) {
    for (std::size_t j = 0; j < i; ++j) {
      const auto& a = source.files[i];
      const auto& b = source.files[j];
      if (a.name == b.name || (!a.path.empty() && a.path == b.path) ||
          (a.inode != 0 && a.device == b.device && a.inode == b.inode))
        return std::unexpected(Bad("indexed source file alias"));
    }
  }
  IndexedPartPlan plan;
  plan.cost_ = *bound;
  plan.source_ = std::move(source);
  const auto& tensors = plan.source_.tensors;
  plan.names_.resize(tensors.size());
  std::iota(plan.names_.begin(), plan.names_.end(), std::uint32_t{0});
  std::ranges::sort(plan.names_, [&](auto a, auto b) { return tensors[a].name < tensors[b].name; });
  for (std::size_t i = 1; i < plan.names_.size(); ++i)
    if (tensors[plan.names_[i - 1]].name == tensors[plan.names_[i]].name)
      return std::unexpected(Bad("duplicate indexed source name"));
  std::vector<std::uint32_t> extents(tensors.size());
  std::iota(extents.begin(), extents.end(), std::uint32_t{0});
  for (const auto& t : tensors) {
    if (t.extent.file >= plan.source_.files.size() || t.extent.bytes == 0 ||
        t.extent.offset > plan.source_.files[t.extent.file].bytes ||
        t.extent.bytes > plan.source_.files[t.extent.file].bytes - t.extent.offset)
      return std::unexpected(Bad("indexed source extent outside file"));
    auto parts = format::PartsOf(t.format);
    auto part = parts ? parts->Find(t.part) : std::nullopt;
    if (!part || part->bytes != t.extent.bytes)
      return std::unexpected(Bad("indexed source part does not match format"));
  }
  std::ranges::sort(extents, [&](auto a, auto b) {
    return std::tie(tensors[a].extent.file, tensors[a].extent.offset) <
           std::tie(tensors[b].extent.file, tensors[b].extent.offset);
  });
  for (std::size_t i = 1; i < extents.size(); ++i) {
    const auto& a = tensors[extents[i - 1]].extent;
    const auto& b = tensors[extents[i]].extent;
    if (a.file == b.file && a.offset + a.bytes > b.offset)
      return std::unexpected(Bad("overlapping indexed source extents"));
  }
  std::vector<std::uint32_t> output_names(outputs.size());
  std::iota(output_names.begin(), output_names.end(), std::uint32_t{0});
  std::ranges::sort(output_names,
                    [&](auto a, auto b) { return outputs[a].name < outputs[b].name; });
  for (std::size_t i = 1; i < output_names.size(); ++i)
    if (outputs[output_names[i - 1]].name == outputs[output_names[i]].name)
      return std::unexpected(Bad("duplicate indexed output name"));
  std::vector<std::uint8_t> used(tensors.size(), 0);
  plan.outputs_.reserve(outputs.size());
  plan.members_.reserve(static_cast<std::size_t>(bound->members));
  plan.physical_order_.reserve(static_cast<std::size_t>(bound->members));
  for (std::size_t oi = 0; oi < outputs.size(); ++oi) {
    const auto& out = outputs[oi];
    auto out_parts = format::PartsOf(out.format);
    if (!out_parts || out_parts->size() != out.parts.size())
      return std::unexpected(Bad("indexed output part coverage is incomplete"));
    const auto basis = std::ranges::find_if(
        out.parts, [](const auto& p) { return p.part != format::PartRole::kTensorScale; });
    if (basis == out.parts.end())
      return std::unexpected(Bad("indexed output lacks a non-scalar part"));
    for (const auto& part : out.parts) {
      const bool shared = (basis->kind == PartAssemblyKind::kRows ||
                           basis->kind == PartAssemblyKind::kInterleavedRows) &&
                          part.part == format::PartRole::kTensorScale;
      if (part.kind != (shared ? PartAssemblyKind::kSharedTensorScale : basis->kind) ||
          (!shared && part.sources.size() != basis->sources.size()))
        return std::unexpected(Bad("indexed multipart assembly disagrees on grouping"));
      if (!shared) {
        for (std::size_t i = 0; i < part.sources.size(); ++i) {
          if (part.sources[i] >= tensors.size() || basis->sources[i] >= tensors.size() ||
              tensors[part.sources[i]].format != tensors[basis->sources[i]].format)
            return std::unexpected(Bad("indexed multipart member geometry disagrees"));
        }
      }
    }
    std::array<bool, format::PartList::kMax> seen{};
    for (const auto& assembly : out.parts) {
      std::size_t pi = 0;
      while (pi < out_parts->size() && (*out_parts)[pi].role != assembly.part) ++pi;
      if (pi == out_parts->size() || seen[pi])
        return std::unexpected(Bad("duplicate or unknown indexed output part"));
      seen[pi] = true;
      const auto kind = assembly.kind;
      const bool single = kind == PartAssemblyKind::kCopy ||
                          kind == PartAssemblyKind::kDenseReshape ||
                          kind == PartAssemblyKind::kSharedTensorScale;
      const bool relaid = out.relayout_from.has_value();
      // A folded output's permutation reorders its source.
      auto unfolded = out.format;
      if (out.fold) {
        if (assembly.sources.empty() || assembly.sources.front() >= tensors.size())
          return std::unexpected(Bad("indexed fold source missing"));
        unfolded.encoding = tensors[assembly.sources.front()].format.encoding;
        if (!IsFoldSourceEncoding(unfolded.encoding))
          return std::unexpected(Bad("invalid indexed fold source encoding"));
      }
      const bool interleaved = kind == PartAssemblyKind::kInterleavedRows;
      const bool atom_table = relaid && out.format.layout == format::Layout::kScaleAtoms;
      const bool row_table = relaid && out.format.layout == format::Layout::kRowInterleaved;
      if (relaid &&
          (out.permutation || out.fold || *out.relayout_from != format::Layout::kSplit ||
           (atom_table && (kind != PartAssemblyKind::kMatrices || out.format.params.tensor_scale ||
                           out.format.shape.rank != 3 || out.format.stacked == 0 ||
                           (assembly.part != format::PartRole::kCodes &&
                            assembly.part != format::PartRole::kBlockScales))) ||
           (row_table && (out.format.shape.rank != 2 || out.format.stacked != 1 ||
                          (assembly.part == format::PartRole::kData
                               ? !interleaved
                               : (assembly.part != format::PartRole::kTensorScale ||
                                  kind != PartAssemblyKind::kSharedTensorScale)))) ||
           (!atom_table && !row_table)))
        return std::unexpected(Bad("invalid indexed relayout"));
      if (interleaved && (!row_table || assembly.sources.size() % 2 != 0))
        return std::unexpected(Bad("invalid indexed row interleave"));
      if (kind == PartAssemblyKind::kScalars &&
          (out.permutation || out.fold || relaid || out.parts.size() != 1 ||
           assembly.part != format::PartRole::kData ||
           out.format.encoding != format::Encoding::kF32 ||
           out.format.layout != format::Layout::kDense || out.format.shape.rank != 1 ||
           out.format.shape.ne[0] != assembly.sources.size()))
        return std::unexpected(Bad("invalid indexed scalar gather"));
      if ((single && assembly.sources.size() != 1) ||
          (out.permutation &&
           (kind != PartAssemblyKind::kCopy && kind != PartAssemblyKind::kDenseReshape)) ||
          (out.permutation && out.permutation->format() != unfolded) ||
          (out.fold && (kind != PartAssemblyKind::kCopy || out.parts.size() != 1 ||
                        out.format.layout != format::Layout::kDense ||
                        out.format.encoding != format::Encoding::kF32)))
        return std::unexpected(Bad("invalid indexed assembly transform"));
      std::uint64_t destination = 0, rows = 0, ordinal = 0;
      // An interleaved table's row stride: a row's codes and its scales.
      std::optional<std::uint64_t> stride;
      if (interleaved)
        if (auto bytes = format::RowBytes(out.format, format::PartRole::kData)) stride = *bytes;
      if (interleaved && (!stride || *stride == 0))
        return std::unexpected(Bad("invalid indexed row interleave"));
      for (auto si : assembly.sources) {
        if (si >= tensors.size() || used[si] != 0 ||
            dispositions[si].kind != SourceDispositionKind::kRouted || dispositions[si].rule != 0)
          return std::unexpected(Bad("indexed source missing, repeated or excluded"));
        const auto& t = tensors[si];
        // A relayout's sources: the output's encoding in the source layout,
        // one matrix each (their tensor scales gathered elsewhere).
        // A fold's source becomes its F32 output, the shape unchanged.
        auto as_output = t.format;
        if (out.fold && IsFoldSourceEncoding(t.format.encoding))
          as_output.encoding = format::Encoding::kF32;
        if (atom_table) {
          as_output.layout = out.format.layout;
          as_output.stacked = out.format.stacked;
          as_output.params.tensor_scale = false;
        }
        const bool scalar = kind == PartAssemblyKind::kScalars;
        // An interleaved pair: a block's codes, then its scales.
        const auto lane_part =
            ordinal % 2 == 0 ? format::PartRole::kCodes : format::PartRole::kBlockScales;
        if (row_table) as_output.layout = out.format.layout;
        if (scalar ? (t.part != format::PartRole::kTensorScale || t.extent.bytes != 4 ||
                      !t.format.params.tensor_scale || t.format.stacked != 1)
                   : ((interleaved ? t.part != lane_part : t.part != assembly.part) ||
                      !SameEncoding(as_output, out.format) ||
                      (relaid && (t.format.layout != *out.relayout_from || t.format.stacked != 1))))
          return std::unexpected(Bad("indexed assembly changes part or encoding"));
        bool valid = false;
        switch (kind) {
          case PartAssemblyKind::kCopy:
            valid = out.fold ? (IsFoldSourceEncoding(t.format.encoding) &&
                                out.format.encoding == format::Encoding::kF32 &&
                                SameEncoding(as_output, out.format) &&
                                t.format.layout == format::Layout::kDense &&
                                as_output.shape == out.format.shape)
                             : t.format == out.format;
            break;
          case PartAssemblyKind::kDenseReshape:
            valid = DenseReshape(t.format, out.format);
            break;
          case PartAssemblyKind::kRows:
            valid = GatherEncoding(out.format) && out.format.stacked == 1 &&
                    out.format.shape.rank == 2 && t.format.shape.rank == 2 &&
                    t.format.shape.ne[0] == out.format.shape.ne[0] &&
                    assembly.part != format::PartRole::kTensorScale &&
                    Add(rows, t.format.shape.ne[1]);
            break;
          case PartAssemblyKind::kMatrices:
            valid =
                relaid
                    ? (t.format.shape.rank == 2 && t.format.shape.ne[0] == out.format.shape.ne[0] &&
                       t.format.shape.ne[1] * out.format.stacked == out.format.shape.ne[1] &&
                       out.format.shape.ne[2] * out.format.stacked == assembly.sources.size())
                    : (GatherEncoding(out.format) && out.format.stacked == 1 &&
                       out.format.shape.rank == 3 && t.format.shape.rank == 2 &&
                       t.format.shape.ne[0] == out.format.shape.ne[0] &&
                       t.format.shape.ne[1] == out.format.shape.ne[1] &&
                       out.format.shape.ne[2] == assembly.sources.size());
            break;
          case PartAssemblyKind::kScalars:
            valid = true;  // checked above: a 4-byte tensor scale each
            break;
          case PartAssemblyKind::kInterleavedRows:
            valid = t.format.shape.rank == 2 && t.format.shape.ne[0] == out.format.shape.ne[0] &&
                    (ordinal % 2 == 0 ? Add(rows, t.format.shape.ne[1])
                                      : tensors[assembly.sources[ordinal - 1]].format == t.format);
            break;
          case PartAssemblyKind::kSharedTensorScale:
            valid = out.format.encoding == format::Encoding::kNvfp4 && out.format.shape.rank == 2 &&
                    t.format.shape.rank == 2 && out.format.stacked == 1 &&
                    t.format.shape.ne[0] == out.format.shape.ne[0] &&
                    assembly.part == format::PartRole::kTensorScale && t.extent.bytes == 4;
            break;
        }
        if (!valid) return std::unexpected(Bad("indexed assembly geometry mismatch"));
        const bool atoms = atom_table && assembly.part == format::PartRole::kBlockScales;
        const auto block = format::Info(t.format.encoding).block_elements;
        std::optional<std::uint64_t> row_bytes;
        if (interleaved)
          if (auto bytes = format::RowBytes(t.format, t.part)) row_bytes = *bytes;
        if (interleaved && !row_bytes)
          return std::unexpected(Bad("invalid indexed row interleave"));
        const std::uint64_t elem_size = (t.format.encoding == format::Encoding::kF32) ? 4U : 2U;
        auto route =
            interleaved ? PartRoutePlan::Strided(t.format.shape.ne[1], *row_bytes, *stride,
                                                 ordinal % 2 == 0 ? 0 : *stride - *row_bytes)
            : out.permutation && out.fold
                ? PartRoutePlan::PermuteFold(*out.permutation, assembly.part, *out.fold)
            : out.permutation ? PartRoutePlan::Permute(*out.permutation, assembly.part)
            : out.fold
                ? PartRoutePlan::Fold(*out.fold, t.extent.bytes / elem_size, t.format.encoding)
            : atoms ? PartRoutePlan::ScaleAtoms(t.format.shape.ne[1],
                                                block == 0 ? 0 : t.format.shape.ne[0] / block)
                    : PartRoutePlan::Copy(t.extent.bytes);
        const std::uint64_t produced =
            out.fold ? (t.extent.bytes / elem_size) * 4U : t.extent.bytes;
        // The output range it writes in (an interleaved lane's rows' span).
        const std::uint64_t span = route ? route->span() : 0;
        if (!route || route->input_bytes() != t.extent.bytes || route->output_bytes() != produced ||
            destination > (*out_parts)[pi].bytes || span > (*out_parts)[pi].bytes - destination)
          return std::unexpected(Bad("indexed assembly byte coverage mismatch"));
        plan.members_.push_back({.source = si,
                                 .output = static_cast<std::uint32_t>(oi),
                                 .part = assembly.part,
                                 .extent = t.extent,
                                 .destination = destination,
                                 .route = *route});
        // An interleaved pair's codes and scales share their rows' span.
        if (!interleaved || ordinal % 2 == 1) destination += span;
        used[si] = 1;
        ++ordinal;
      }
      if (destination != (*out_parts)[pi].bytes ||
          ((kind == PartAssemblyKind::kRows || interleaved) && rows != out.format.shape.ne[1]))
        return std::unexpected(Bad("indexed assembly does not cover output part"));
    }
    plan.outputs_.push_back({.name = out.name,
                             .format = out.format,
                             .parts = *out_parts,
                             .relayout_from = out.relayout_from});
  }
  for (std::size_t i = 0; i < dispositions.size(); ++i) {
    const auto& d = dispositions[i];
    if ((d.kind == SourceDispositionKind::kRouted && (used[i] == 0 || d.rule != 0)) ||
        ((d.kind == SourceDispositionKind::kSemantic ||
          d.kind == SourceDispositionKind::kAuxiliary) &&
         (used[i] != 0 || d.rule == 0)) ||
        (d.kind != SourceDispositionKind::kRouted && d.kind != SourceDispositionKind::kSemantic &&
         d.kind != SourceDispositionKind::kAuxiliary))
      return std::unexpected(Bad("indexed source disposition is incomplete"));
  }
  plan.dispositions_.assign(dispositions.begin(), dispositions.end());
  plan.physical_order_.resize(plan.members_.size());
  std::iota(plan.physical_order_.begin(), plan.physical_order_.end(), std::uint32_t{0});
  std::ranges::sort(plan.physical_order_, [&](auto a, auto b) {
    return std::tie(plan.members_[a].extent.file, plan.members_[a].extent.offset) <
           std::tie(plan.members_[b].extent.file, plan.members_[b].extent.offset);
  });
  return plan;
}

std::optional<std::uint32_t> IndexedPartPlan::FindSource(std::string_view name) const {
  auto at = std::lower_bound(names_.begin(), names_.end(), name,
                             [&](auto i, auto n) { return source_.tensors[i].name < n; });
  if (at == names_.end() || source_.tensors[*at].name != name) return std::nullopt;
  return *at;
}
}  // namespace llmp::import
