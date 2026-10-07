// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "model/gemma3.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <initializer_list>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "artifact/representation.h"
#include "base/check.h"
#include "base/sha256.h"

namespace jitllm::model {
namespace {
constexpr std::uint64_t kExtent = 2U << 20U;
constexpr std::uint16_t kZero = 0, kNegInf = 0xFC00;
std::unexpected<std::string> Refused(std::string detail) {
  return std::unexpected(std::move(detail));
}
std::uint64_t Pad(std::uint64_t bytes, std::uint64_t align) {
  return ((bytes + align - 1) / align) * align;
}
bool ProfileValid(const Gemma3Profile& p) { return p == Gemma3_4BQat(); }
// Used only with this fixed profile's expected shapes. Equality precedes
// multiplication, so untrusted dimensions never enter the byte calculation.
std::optional<std::uint64_t> TensorBytes(std::string_view type,
                                         std::span<const std::uint64_t> dimensions,
                                         std::string_view expected,
                                         std::initializer_list<std::uint64_t> shape) {
  const auto* traits = artifact::FindGgmlType(expected);
  if (traits == nullptr || type != expected || !std::ranges::equal(dimensions, shape) ||
      shape.size() == 0 || dimensions[0] % traits->block_elements != 0)
    return std::nullopt;
  auto bytes = dimensions[0] / traits->block_elements * traits->block_bytes;
  for (std::size_t dim = 1; dim < dimensions.size(); ++dim) bytes *= dimensions[dim];
  return bytes;
}
bool AlignValid(std::uint32_t align) { return align >= 256 && align <= 8192 && align % 256 == 0; }
std::uint32_t ReadCells(std::uint32_t positions, std::uint32_t cells, std::uint32_t align) {
  return static_cast<std::uint32_t>(std::min<std::uint64_t>(cells, Pad(positions, align)));
}
// Public layouts are checked too: an edited offset/width must never produce
// out-of-region input indices or understate materialization ranges.
bool LayoutValid(const Gemma3Profile& p, const Gemma3StateLayout& s) {
  if (!ProfileValid(p) || s.context == 0 || s.context > p.context || s.max_rows == 0 ||
      s.max_rows > std::min(s.context, kGemma3MaxRows) || s.global_cells != Pad(s.context, 256) ||
      s.local_cells != Pad(std::min<std::uint64_t>(s.context, p.window + s.max_rows), 256) ||
      s.tensors.size() != std::size_t{p.layers} * 2) {
    return false;
  }
  std::uint64_t at = 0;
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    for (const bool value : {false, true}) {
      const auto& t = s.tensors[std::size_t{il} * 2 + (value ? 1U : 0U)];
      const auto cells = p.local(il) ? s.local_cells : s.global_cells;
      const auto width = (value ? p.value_dim : p.key_dim) * p.kv_heads;
      const auto bytes = std::uint64_t{width} * cells * 2;
      if (t.layer != il || t.value != value || t.local != p.local(il) || t.width != width ||
          t.cells != cells || t.offset != at || t.bytes != bytes) {
        return false;
      }
      at += Pad(bytes, kExtent);
    }
  }
  return s.bytes == at;
}
std::expected<std::uint64_t, std::string> InputBytes(const Gemma3Profile& p,
                                                     const Gemma3StateLayout& s,
                                                     std::span<const Gemma3Segment> segments,
                                                     bool masks, std::uint32_t align) {
  if (!LayoutValid(p, s) || !AlignValid(align) || segments.empty() ||
      segments.size() > kGemma3MaxSlots) {
    return Refused("invalid Gemma3 layout, alignment or segment count");
  }
  std::array<bool, kGemma3MaxSlots> seen{};
  std::uint64_t bytes = segments.size() * sizeof(Gemma3SegmentInputs), rows = 0;
  for (const auto& seg : segments) {
    if (seg.slot >= kGemma3MaxSlots || seen[seg.slot] || seg.tokens.empty() ||
        seg.n_past > s.context || seg.tokens.size() > s.context - seg.n_past ||
        seg.tokens.size() > s.max_rows - rows) {
      return Refused("empty, repeated or out-of-bounds Gemma3 segment");
    }
    seen[seg.slot] = true;
    for (const auto token : seg.tokens) {
      if (token < 0 || std::cmp_greater_equal(token, p.vocab)) {
        return Refused("Gemma3 token is outside the vocabulary");
      }
    }
    rows += seg.tokens.size();
    // Token, position, output ID and the global/local I64 cell indices.
    bytes += seg.tokens.size() * (3 * sizeof(std::int32_t) + 2 * sizeof(std::int64_t));
    if (masks) {
      const auto end = seg.n_past + static_cast<std::uint32_t>(seg.tokens.size());
      const auto global =
          std::uint64_t{ReadCells(end, s.global_cells, align)} * seg.tokens.size() * 2;
      const auto local =
          std::uint64_t{ReadCells(end, s.local_cells, align)} * seg.tokens.size() * 2;
      if (global > std::numeric_limits<std::int32_t>::max() ||
          local > std::numeric_limits<std::int32_t>::max()) {
        return Refused("Gemma3 reference mask exceeds the GGML I32 byte stride");
      }
      bytes += global + local;
    }
  }
  return bytes;
}
}  // namespace

const Gemma3Profile& Gemma3_4BQat() {
  // Facts from bbcac0d0's approved Q4_0 file, not a Gemma4-derived profile.
  static constexpr Gemma3Profile profile{
      34, 2560, 10240, 8, 4, 256, 256, 262208, 131072, 1024, 1000000.0F, 8.0F, 1.0e-6F, "linear"};
  return profile;
}

std::expected<Gemma3Binding, std::string> BindApprovedGemma3(const artifact::Artifact& artifact) {
  const auto sources = artifact.sources();
  if (sources.size() != 1 || sources[0].name != "gemma-3-4b-it-qat-Q4_0.gguf" ||
      sources[0].bytes.value() != 2526080992ULL ||
      base::ToHex(sources[0].sha256) !=
          "ee91c3e7a4ab95d8c95672f9fcb58bf236b257e9f217966bcf53a5a6df4ab49a")
    return std::unexpected("Gemma3 execution needs the approved prepared QAT source identity");
  return BindGemma3(Gemma3_4BQat(), artifact);
}
std::expected<Gemma3Binding, std::string> BindGemma3(const Gemma3Profile& p,
                                                     std::string_view architecture,
                                                     std::span<const Gemma3Resource> resources) {
  if (p != Gemma3_4BQat() || architecture != "gemma3" || resources.size() != 444) {
    return std::unexpected("unsupported Gemma3 profile, architecture or resource count");
  }
  std::unordered_map<std::string, std::uint32_t> roles;
  for (std::size_t i = 0; i < resources.size(); ++i) {
    const auto& resource = resources[i];
    if (resource.roles.empty() || resource.roles.size() > 2) {
      return std::unexpected("invalid Gemma3 resource roles");
    }
    for (const auto& role : resource.roles) {
      if (role.empty() || !roles.emplace(role, static_cast<std::uint32_t>(i)).second) {
        return std::unexpected("empty or duplicate Gemma3 role");
      }
    }
  }
  const auto take =
      [&](const std::string& role, std::string_view type,
          std::initializer_list<std::uint64_t> shape) -> std::expected<Gemma3Tensor, std::string> {
    const auto found = roles.find(role);
    if (found == roles.end()) {
      return std::unexpected("missing Gemma3 role: " + role);
    }
    const auto index = found->second;
    const auto& resource = resources[index];
    const auto bytes = TensorBytes(resource.type, resource.ne, type, shape);
    if (!bytes) {
      return std::unexpected("wrong Gemma3 type or shape: " + role);
    }
    // Every approved row needs no additional GGML row padding; readable
    // storage may exceed its canonical bytes without changing identity.
    if (resource.readable < *bytes) {
      return std::unexpected("short Gemma3 readable storage: " + role);
    }
    roles.erase(found);
    return Gemma3Tensor{index, resource.type, resource.ne, resource.readable};
  };
  Gemma3Binding binding;
  const auto assign =
      [&](Gemma3Tensor& target, const std::string& role, std::string_view type,
          std::initializer_list<std::uint64_t> shape) -> std::expected<void, std::string> {
    auto tensor = take(role, type, shape);
    if (!tensor) {
      return std::unexpected(tensor.error());
    }
    target = std::move(*tensor);
    return {};
  };
  if (auto result = assign(binding.token_embd, "token_embd.weight", "Q8_0", {p.width, p.vocab});
      !result) {
    return std::unexpected(result.error());
  }
  binding.output = binding.token_embd;
  if (const auto head = roles.find("output.weight"); head != roles.end()) {
    if (head->second != binding.token_embd.index) {
      return std::unexpected("Gemma3 output must alias token embeddings");
    }
    roles.erase(head);
  }
  if (auto result = assign(binding.output_norm, "output_norm.weight", "F32", {p.width}); !result) {
    return std::unexpected(result.error());
  }
  binding.layers.resize(p.layers);
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    auto& layer = binding.layers[il];
    const auto prefix = "blk." + std::to_string(il) + '.';
    const auto norm = [&](Gemma3Tensor& tensor, std::string_view suffix, std::uint64_t width) {
      return assign(tensor, prefix + std::string(suffix) + ".weight", "F32", {width});
    };
    const auto matrix = [&](Gemma3Tensor& tensor, std::string_view suffix, std::uint64_t input,
                            std::uint64_t output) {
      return assign(tensor, prefix + std::string(suffix) + ".weight", "Q4_0", {input, output});
    };
    // V is a separate actual resource in every layer; do not inherit
    // Gemma4's global-layer K/V identity rule.
    for (auto result :
         {norm(layer.attn_norm, "attn_norm", p.width),
          matrix(layer.q, "attn_q", p.width, p.heads * p.key_dim),
          matrix(layer.k, "attn_k", p.width, p.kv_heads * p.key_dim),
          matrix(layer.v, "attn_v", p.width, p.kv_heads * p.value_dim),
          matrix(layer.out, "attn_output", p.heads * p.value_dim, p.width),
          norm(layer.q_norm, "attn_q_norm", p.key_dim),
          norm(layer.k_norm, "attn_k_norm", p.key_dim),
          norm(layer.attn_post_norm, "post_attention_norm", p.width),
          norm(layer.ffn_norm, "ffn_norm", p.width), matrix(layer.gate, "ffn_gate", p.width, p.ffn),
          matrix(layer.up, "ffn_up", p.width, p.ffn),
          matrix(layer.down, "ffn_down", p.ffn, p.width),
          norm(layer.ffn_post_norm, "post_ffw_norm", p.width)}) {
      if (!result) {
        return std::unexpected(result.error());
      }
    }
  }
  if (!roles.empty()) {
    return std::unexpected("unconsumed Gemma3 roles");
  }
  return binding;
}

std::expected<Gemma3Binding, std::string> BindGemma3(const Gemma3Profile& p,
                                                     const artifact::Artifact& artifact) {
  if (artifact.model().expert_count != 0 || !artifact.expert_arrays().empty() ||
      artifact.resources().size() != 444) {
    return std::unexpected("Gemma3 requires 444 ordinary resources and no expert storage");
  }
  std::vector<Gemma3Resource> resources;
  resources.reserve(artifact.resources().size());
  for (const auto& resource : artifact.resources()) {
    if (resource.repr.family != artifact::Family::kGgml) {
      return std::unexpected("Gemma3 requires GGML resources");
    }
    resources.push_back({resource.roles, std::string(resource.repr.type), resource.repr.dims,
                         resource.readable.value()});
  }
  return BindGemma3(p, artifact.model().architecture, resources);
}
std::expected<void, std::string> CheckGemma3Binding(const Gemma3Profile& p,
                                                    const Gemma3Binding& binding) {
  if (!ProfileValid(p) || binding.layers.size() != p.layers ||
      binding.output != binding.token_embd) {
    return Refused("invalid Gemma3 profile, layer count or tied head");
  }
  // Every non-head role owns one of exactly 444 resource identities. The
  // tied output was checked above; no other alias is valid. Direct fixed-shape
  // checks preserve public mutation refusal without rebuilding role strings,
  // resource vectors, a role map and a second binding on every source check.
  std::array<bool, 444> seen{};
  const auto check = [&](const Gemma3Tensor& tensor, std::string_view type,
                         std::initializer_list<std::uint64_t> shape) {
    if (tensor.index >= seen.size() || seen[tensor.index]) return false;
    const auto bytes = TensorBytes(tensor.type, tensor.ne, type, shape);
    if (!bytes || tensor.readable < *bytes) return false;
    seen[tensor.index] = true;
    return true;
  };
  if (!check(binding.token_embd, "Q8_0", {p.width, p.vocab}) ||
      !check(binding.output_norm, "F32", {p.width}))
    return Refused("invalid Gemma3 embedding or output norm descriptor");
  for (const auto& layer : binding.layers) {
    if (!check(layer.attn_norm, "F32", {p.width}) ||
        !check(layer.q, "Q4_0", {p.width, p.heads * p.key_dim}) ||
        !check(layer.k, "Q4_0", {p.width, p.kv_heads * p.key_dim}) ||
        !check(layer.v, "Q4_0", {p.width, p.kv_heads * p.value_dim}) ||
        !check(layer.out, "Q4_0", {p.heads * p.value_dim, p.width}) ||
        !check(layer.q_norm, "F32", {p.key_dim}) || !check(layer.k_norm, "F32", {p.key_dim}) ||
        !check(layer.attn_post_norm, "F32", {p.width}) ||
        !check(layer.ffn_norm, "F32", {p.width}) || !check(layer.gate, "Q4_0", {p.width, p.ffn}) ||
        !check(layer.up, "Q4_0", {p.width, p.ffn}) ||
        !check(layer.down, "Q4_0", {p.ffn, p.width}) ||
        !check(layer.ffn_post_norm, "F32", {p.width}))
      return Refused("invalid Gemma3 layer descriptor or repeated resource identity");
  }
  return {};
}

std::expected<Gemma3StateLayout, std::string> Gemma3State(const Gemma3Profile& p,
                                                          std::uint32_t context,
                                                          std::uint32_t max_rows) {
  if (!ProfileValid(p) || context == 0 || context > p.context || max_rows == 0 ||
      max_rows > std::min(context, kGemma3MaxRows)) {
    return Refused("Gemma3 context or chunk bound is invalid");
  }
  Gemma3StateLayout s;
  s.context = context;
  s.max_rows = max_rows;
  s.global_cells = static_cast<std::uint32_t>(Pad(context, 256));
  s.local_cells =
      static_cast<std::uint32_t>(Pad(std::min<std::uint64_t>(context, p.window + max_rows), 256));
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    for (const bool value : {false, true}) {
      const auto width = (value ? p.value_dim : p.key_dim) * p.kv_heads;
      const auto cells = p.local(il) ? s.local_cells : s.global_cells;
      const auto bytes = std::uint64_t{width} * cells * 2;
      s.tensors.push_back({.layer = il,
                           .value = value,
                           .local = p.local(il),
                           .width = width,
                           .cells = cells,
                           .offset = s.bytes,
                           .bytes = bytes});
      s.bytes += Pad(bytes, kExtent);
    }
  }
  return s;
}
std::expected<std::vector<StateRepresentation>, std::string> Gemma3StateLayout::Representations(
    const Gemma3Profile& p) const {
  if (!LayoutValid(p, *this)) {
    return Refused("Gemma3 representation layout is invalid");
  }
  std::vector<StateRepresentation> out;
  for (const auto& t : tensors) {
    out.push_back({.name = std::format("gemma3.{}.{}", t.layer, t.value ? "v" : "k"),
                   .block_positions = 0,
                   .block_bytes = base::Bytes{Pad(t.bytes, kExtent)},
                   .capabilities = t.local ? static_cast<std::uint8_t>(StateCapability::kAppend)
                                           : StateCapability::kAppend | StateCapability::kTruncate,
                   .max_snapshots = 0,
                   .snapshot_bytes = base::Bytes{0}});
  }
  return out;
}
std::expected<std::vector<StateRange>, std::string> Gemma3UsedState(const Gemma3Profile& p,
                                                                    const Gemma3StateLayout& s,
                                                                    std::uint32_t positions,
                                                                    std::uint32_t align) {
  if (!LayoutValid(p, s) || !AlignValid(align) || positions > s.context) {
    return Refused("Gemma3 initialized footprint is out of bounds");
  }
  std::vector<StateRange> out;
  if (positions != 0) {
    for (const auto& t : s.tensors) {
      out.push_back({.offset = t.offset,
                     .bytes = std::uint64_t{ReadCells(positions, t.cells, align)} * t.width * 2});
    }
  }
  return out;
}
std::expected<std::uint64_t, std::string> Gemma3HostInputBytes(
    const Gemma3Profile& p, const Gemma3StateLayout& s, std::span<const Gemma3Segment> segments,
    bool masks, std::uint32_t align) {
  return InputBytes(p, s, segments, masks, align);
}
std::expected<Gemma3ChunkInputs, std::string> Gemma3Chunk(const Gemma3Profile& p,
                                                          const Gemma3StateLayout& s,
                                                          std::span<const Gemma3Segment> segments,
                                                          bool masks, std::uint32_t align) {
  const auto bytes = InputBytes(p, s, segments, masks, align);
  if (!bytes) {
    return std::unexpected(bytes.error());
  }
  Gemma3ChunkInputs in;
  std::size_t total_rows = 0;
  for (const auto& seg : segments) {
    total_rows += seg.tokens.size();
  }
  in.tokens.reserve(total_rows);
  in.positions.reserve(total_rows);
  in.out_ids.reserve(total_rows);
  in.segments.reserve(segments.size());
  for (const auto& seg : segments) {
    Gemma3SegmentInputs x;
    x.slot = seg.slot;
    x.n_past = seg.n_past;
    x.first_row = static_cast<std::uint32_t>(in.tokens.size());
    x.rows = static_cast<std::uint32_t>(seg.tokens.size());
    const auto end = x.n_past + x.rows;
    x.global_n_kv = ReadCells(end, s.global_cells, align);
    x.local_n_kv = ReadCells(end, s.local_cells, align);
    x.global_cells.resize(x.rows);
    x.local_cells.resize(x.rows);
    if (masks) {
      x.global_mask.assign(std::size_t{x.rows} * x.global_n_kv, kNegInf);
      x.local_mask.assign(std::size_t{x.rows} * x.local_n_kv, kNegInf);
    }
    for (std::uint32_t i = 0; i < x.rows; ++i) {
      const auto pos = x.n_past + i;
      in.tokens.push_back(seg.tokens[i]);
      in.positions.push_back(static_cast<std::int32_t>(pos));
      in.out_ids.push_back(static_cast<std::int32_t>(in.out_ids.size()));
      x.global_cells[i] = pos;
      x.local_cells[i] = pos % s.local_cells;
      if (masks) {
        for (std::uint32_t cell = 0; cell < x.global_n_kv; ++cell) {
          if (cell <= pos) {
            x.global_mask[std::size_t{i} * x.global_n_kv + cell] = kZero;
          }
        }
        for (std::uint32_t cell = 0; cell < x.local_n_kv; ++cell) {
          if (cell >= end) {
            continue;
          }
          const auto held = cell + ((end - 1 - cell) / s.local_cells) * s.local_cells;
          if (held <= pos && pos - held < p.window) {
            x.local_mask[std::size_t{i} * x.local_n_kv + cell] = kZero;
          }
        }
      }
    }
    in.segments.push_back(std::move(x));
  }
  std::uint64_t allocated =
      (in.tokens.capacity() + in.positions.capacity() + in.out_ids.capacity()) *
          sizeof(std::int32_t) +
      in.segments.capacity() * sizeof(Gemma3SegmentInputs);
  for (const auto& seg : in.segments) {
    allocated += (seg.global_cells.capacity() + seg.local_cells.capacity()) * sizeof(std::int64_t);
    allocated += (seg.global_mask.capacity() + seg.local_mask.capacity()) * sizeof(std::uint16_t);
  }
  base::Check(allocated == *bytes, "Gemma3 inputs exceeded their preflight storage envelope");
  return in;
}
std::expected<std::vector<StateRange>, std::string> Gemma3ChunkWrites(const Gemma3Profile& p,
                                                                      const Gemma3StateLayout& s,
                                                                      std::uint32_t past,
                                                                      std::uint32_t rows) {
  if (!LayoutValid(p, s) || rows == 0 || rows > s.max_rows || past > s.context ||
      rows > s.context - past) {
    return Refused("Gemma3 write footprint is out of bounds");
  }
  std::vector<StateRange> out;
  for (const auto& t : s.tensors) {
    const auto first = t.local ? past % t.cells : past;
    const auto count = std::min(rows, t.cells - first);
    const auto row_bytes = std::uint64_t{t.width} * 2;
    out.push_back({.offset = t.offset + first * row_bytes, .bytes = count * row_bytes});
    if (count < rows) {
      out.push_back({.offset = t.offset, .bytes = (rows - count) * row_bytes});
    }
  }
  return out;
}
}  // namespace jitllm::model
