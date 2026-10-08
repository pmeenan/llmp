// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// Dsv4CompressorPlan ports llama.cpp b29c606e2's dsv4_build_comp_plan
// (src/llama-kv-cache-dsv4.cpp, MIT) for one sequence, so the file carries
// GGML's notice (docs/licensing.md).

#include "model/dsv4.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "artifact/representation.h"
#include "model/host_mask.h"
#include "model/state.h"

namespace llmp::model {
namespace {

std::unexpected<std::string> Refused(std::string detail) {
  return std::unexpected(std::move(detail));
}

std::uint64_t Pad(std::uint64_t n, std::uint64_t to) { return (n + to - 1) / to * to; }

// What a bound tensor's type must be.
enum class Kind : std::uint8_t {
  kF32,     // norms, scales, biases, sinks
  kFloat,   // APE lookup tables: F32 or F16, gathered into F32
  kI32,     // the hash layers' token-to-expert tables
  kMatrix,  // anything a product or a lookup reads: F32, F16, BF16 or block-quantized
};

bool IsMatrixType(std::string_view type) {
  return !type.empty() && type != "I8" && type != "I16" && type != "I32" && type != "I64" &&
         type != "F64";
}

bool TypeFits(Kind kind, std::string_view type) {
  switch (kind) {
    case Kind::kF32:
      return type == "F32";
    case Kind::kFloat:
      return type == "F32" || type == "F16";
    case Kind::kI32:
      return type == "I32";
    case Kind::kMatrix:
      return IsMatrixType(type);
  }
  return false;
}

std::string_view KindName(Kind kind) {
  switch (kind) {
    case Kind::kF32:
      return "F32";
    case Kind::kFloat:
      return "F32 or F16";
    case Kind::kI32:
      return "I32";
    case Kind::kMatrix:
      return "a matrix";
  }
  return "a matrix";
}

struct Want {
  std::string role;
  Kind kind = Kind::kMatrix;
  std::vector<std::uint64_t> ne;
  bool expert_array = false;
  Dsv4Tensor* into = nullptr;
};

std::string Shape(std::span<const std::uint64_t> ne) {
  std::string out = "[";
  for (std::size_t i = 0; i < ne.size(); ++i) {
    out += std::format("{}{}", i == 0 ? "" : ", ", ne[i]);
  }
  return out + "]";
}

bool ProfileIsSane(const Dsv4Profile& p) {
  return p.layers > 0 && p.width > 0 && p.heads > 0 && p.head_dim > 0 && p.rope_dims > 0 &&
         p.rope_dims <= p.head_dim && p.rope_dims % 2 == 0 && p.o_groups > 0 &&
         p.heads % p.o_groups == 0 && p.q_lora > 0 && p.o_lora > 0 && p.window > 0 &&
         p.experts > 0 && p.experts_used > 0 && p.experts_used <= p.experts && p.expert_ffn > 0 &&
         p.shared_experts > 0 && p.hc == 4 && p.indexer_heads > 0 &&
         p.indexer_head_dim >= p.rope_dims && p.indexer_top_k > 0 && p.vocab > 0 &&
         p.compress_ratios.size() == p.layers && p.hash_layers <= p.layers &&
         std::ranges::all_of(p.compress_ratios, [](std::uint32_t r) {
           return r == 0 || r == kDsv4CsaRatio || r == kDsv4HcaRatio;
         });
}

}  // namespace

const Dsv4Profile& Dsv4Flash() {
  // deepseek4.* key/values of DeepSeek-V4-Flash-0731-UD-Q2_K_XL (and the
  // e3aa0d6a revision), checked by
  // docs/experiments/dsv4-native/gguf_profile_check.py. The epsilons are
  // the GGUF's float32 values (9.999999974752427e-07).
  static const Dsv4Profile kProfile = [] {
    Dsv4Profile p;
    p.name = "deepseek-v4-flash";
    p.layers = 43;
    p.width = 4096;
    p.heads = 64;
    p.head_dim = 512;
    p.rope_dims = 64;
    p.q_lora = 1024;
    p.o_lora = 1024;
    p.o_groups = 8;
    p.window = 128;
    p.experts = 256;
    p.experts_used = 6;
    p.expert_ffn = 2048;
    p.shared_experts = 1;
    p.hash_layers = 3;
    p.hc = 4;
    p.sinkhorn_iterations = 20;
    p.indexer_heads = 64;
    p.indexer_head_dim = 128;
    p.indexer_top_k = 512;
    p.vocab = 129280;
    p.yarn_original_context = 65536;
    p.rms_eps = 1e-6f;
    p.hc_eps = 1e-6f;
    p.rope_base = 10000.0f;
    p.compress_rope_base = 160000.0f;
    p.rope_scale = 16.0f;
    p.yarn_beta_fast = 32.0f;
    p.yarn_beta_slow = 1.0f;
    p.expert_weights_scale = 1.5f;
    p.expert_weights_norm = true;
    p.swiglu_limit = 10.0f;
    p.swiglu_limit_shared = 10.0f;
    // Layers 0 and 1 are window-only; then CSA and HCA alternate from layer
    // 2, and the last layer is CSA.
    p.compress_ratios.assign(p.layers, 0);
    for (std::uint32_t il = 2; il < p.layers; ++il) {
      p.compress_ratios[il] = il % 2 == 0 ? kDsv4CsaRatio : kDsv4HcaRatio;
    }
    return p;
  }();
  return kProfile;
}

std::expected<Dsv4Binding, std::string> BindDsv4(const Dsv4Profile& p,
                                                 std::string_view architecture,
                                                 std::span<const Dsv4Resource> resources) {
  return BindDsv4Roles(p, "deepseek4", architecture, true, resources, {});
}

std::expected<Dsv4Binding, std::string> BindDsv4Roles(const Dsv4Profile& p,
                                                      std::string_view want_architecture,
                                                      std::string_view architecture, bool tables,
                                                      std::span<const Dsv4Resource> resources,
                                                      std::span<Dsv4ExtraRole> extra) {
  if (architecture != want_architecture) {
    return Refused(
        std::format("the artifact's architecture is {}, not {}", architecture, want_architecture));
  }
  if (!ProfileIsSane(p)) {
    return Refused("the profile is not a DeepSeek V4 model's");
  }
  Dsv4Binding b;
  b.layers.resize(p.layers);
  const std::uint64_t width = p.width;
  const std::uint64_t head = p.head_dim;
  std::vector<Want> want;
  const auto add = [&](std::string role, Kind kind, std::vector<std::uint64_t> ne, Dsv4Tensor* into,
                       bool expert_array = false) {
    want.push_back({.role = std::move(role),
                    .kind = kind,
                    .ne = std::move(ne),
                    .expert_array = expert_array,
                    .into = into});
  };
  if (tables) {
    add("token_embd.weight", Kind::kMatrix, {width, p.vocab}, &b.token_embd);
    add("output.weight", Kind::kMatrix, {width, p.vocab}, &b.output);
  }
  for (Dsv4ExtraRole& e : extra) {
    add(e.role, e.f32 ? Kind::kF32 : Kind::kMatrix, e.ne, e.into);
  }
  add("output_norm.weight", Kind::kF32, {width}, &b.output_norm);
  add("output_hc_fn.weight", Kind::kMatrix, {p.hc_width(), p.hc}, &b.hc_head_fn);
  add("output_hc_base.weight", Kind::kF32, {p.hc}, &b.hc_head_base);
  add("output_hc_scale.weight", Kind::kF32, {1}, &b.hc_head_scale);
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    Dsv4Layer& l = b.layers[il];
    l.ratio = p.compress_ratios[il];
    l.hash = il < p.hash_layers;
    const std::string n = std::format("blk.{}.", il);
    add(n + "attn_norm.weight", Kind::kF32, {width}, &l.attn_norm);
    add(n + "attn_sinks.weight", Kind::kF32, {p.heads}, &l.attn_sinks);
    add(n + "attn_q_a.weight", Kind::kMatrix, {width, p.q_lora}, &l.q_a);
    add(n + "attn_q_a_norm.weight", Kind::kF32, {p.q_lora}, &l.q_a_norm);
    add(n + "attn_q_b.weight", Kind::kMatrix, {p.q_lora, std::uint64_t{p.heads} * head}, &l.q_b);
    add(n + "attn_kv.weight", Kind::kMatrix, {width, head}, &l.kv);
    add(n + "attn_kv_a_norm.weight", Kind::kF32, {head}, &l.kv_norm);
    // Stored as [heads·head / groups, o_lora·groups]; the graph views it as
    // [heads·head / groups, o_lora, groups], as llama.cpp loads it.
    add(n + "attn_output_a.weight", Kind::kMatrix,
        {std::uint64_t{p.heads} * head / p.o_groups, std::uint64_t{p.o_lora} * p.o_groups},
        &l.out_a);
    add(n + "attn_output_b.weight", Kind::kMatrix, {std::uint64_t{p.o_lora} * p.o_groups, width},
        &l.out_b);
    add(n + "hc_attn_fn.weight", Kind::kMatrix, {p.hc_width(), p.hc_mix()}, &l.hc_attn_fn);
    add(n + "hc_attn_base.weight", Kind::kF32, {p.hc_mix()}, &l.hc_attn_base);
    add(n + "hc_attn_scale.weight", Kind::kF32, {3}, &l.hc_attn_scale);
    add(n + "hc_ffn_fn.weight", Kind::kMatrix, {p.hc_width(), p.hc_mix()}, &l.hc_ffn_fn);
    add(n + "hc_ffn_base.weight", Kind::kF32, {p.hc_mix()}, &l.hc_ffn_base);
    add(n + "hc_ffn_scale.weight", Kind::kF32, {3}, &l.hc_ffn_scale);
    if (l.ratio != 0) {
      const std::uint64_t coff = l.ratio == kDsv4CsaRatio ? 2 : 1;
      add(n + "attn_compressor_kv.weight", Kind::kMatrix, {width, coff * head}, &l.comp_kv);
      add(n + "attn_compressor_gate.weight", Kind::kMatrix, {width, coff * head}, &l.comp_gate);
      add(n + "attn_compressor_ape.weight", Kind::kFloat, {coff * head, l.ratio}, &l.comp_ape);
      add(n + "attn_compressor_norm.weight", Kind::kF32, {head}, &l.comp_norm);
    }
    if (l.ratio == kDsv4CsaRatio) {
      const std::uint64_t ih = p.indexer_head_dim;
      add(n + "indexer.attn_q_b.weight", Kind::kMatrix, {p.q_lora, p.indexer_heads * ih},
          &l.idx_q_b);
      add(n + "indexer.proj.weight", Kind::kMatrix, {width, p.indexer_heads}, &l.idx_proj);
      add(n + "indexer_compressor_kv.weight", Kind::kMatrix, {width, 2 * ih}, &l.idx_comp_kv);
      add(n + "indexer_compressor_gate.weight", Kind::kMatrix, {width, 2 * ih}, &l.idx_comp_gate);
      add(n + "indexer_compressor_ape.weight", Kind::kFloat, {2 * ih, kDsv4CsaRatio},
          &l.idx_comp_ape);
      add(n + "indexer_compressor_norm.weight", Kind::kF32, {ih}, &l.idx_comp_norm);
    }
    add(n + "ffn_norm.weight", Kind::kF32, {width}, &l.ffn_norm);
    add(n + "ffn_gate_inp.weight", Kind::kMatrix, {width, p.experts}, &l.router);
    if (l.hash) {
      add(n + "ffn_gate_tid2eid.weight", Kind::kI32, {p.experts_used, p.vocab}, &l.tid2eid);
    } else {
      add(n + "exp_probs_b.bias", Kind::kF32, {p.experts}, &l.router_bias);
    }
    add(n + "ffn_gate_exps.weight", Kind::kMatrix, {width, p.expert_ffn}, &l.gate_exps, true);
    add(n + "ffn_up_exps.weight", Kind::kMatrix, {width, p.expert_ffn}, &l.up_exps, true);
    add(n + "ffn_down_exps.weight", Kind::kMatrix, {p.expert_ffn, width}, &l.down_exps, true);
    const std::uint64_t shared = std::uint64_t{p.expert_ffn} * p.shared_experts;
    add(n + "ffn_gate_shexp.weight", Kind::kMatrix, {width, shared}, &l.gate_shexp);
    add(n + "ffn_up_shexp.weight", Kind::kMatrix, {width, shared}, &l.up_shexp);
    add(n + "ffn_down_shexp.weight", Kind::kMatrix, {shared, width}, &l.down_shexp);
  }

  // Every role the artifact binds, and whether an expert array holds it.
  std::unordered_map<std::string, std::uint32_t> by_role;
  for (std::uint32_t i = 0; i < resources.size(); ++i) {
    for (const std::string& role : resources[i].roles) {
      if (!by_role.emplace(role, i).second) {
        return Refused(std::format("the artifact binds {} twice", role));
      }
    }
  }
  std::unordered_map<std::string_view, const Want*> wanted;
  for (const Want& w : want) {
    wanted.emplace(w.role, &w);
  }
  for (const auto& [role, index] : by_role) {
    if (!wanted.contains(role)) {
      return Refused(std::format("the artifact binds {}, which DeepSeek V4 does not read", role));
    }
  }
  for (const Want& w : want) {
    const auto found = by_role.find(w.role);
    if (found == by_role.end()) {
      return Refused(std::format("the artifact has no {}", w.role));
    }
    const Dsv4Resource& r = resources[found->second];
    if (r.expert_array != w.expert_array) {
      return Refused(std::format("{} is {}an expert array", w.role, r.expert_array ? "" : "not "));
    }
    if (w.expert_array && r.count != p.experts) {
      return Refused(std::format("{} has {} experts, not {}", w.role, r.count, p.experts));
    }
    if (!TypeFits(w.kind, r.type) || !std::ranges::equal(r.ne, w.ne)) {
      return Refused(std::format("{} is {} {}, not {} {}", w.role, r.type, Shape(r.ne),
                                 KindName(w.kind), Shape(w.ne)));
    }
    *w.into = {.index = found->second, .type = r.type, .ne = r.ne};
  }
  return b;
}

std::expected<Dsv4Binding, std::string> BindDsv4(const Dsv4Profile& profile,
                                                 const artifact::Artifact& artifact) {
  return BindDsv4Roles(profile, "deepseek4", artifact, true, {});
}

std::expected<Dsv4Binding, std::string> BindDsv4Roles(const Dsv4Profile& profile,
                                                      std::string_view want_architecture,
                                                      const artifact::Artifact& artifact,
                                                      bool tables, std::span<Dsv4ExtraRole> extra) {
  std::vector<Dsv4Resource> all;
  all.reserve(artifact.resources().size() + artifact.expert_arrays().size());
  for (const artifact::Resource& resource : artifact.resources()) {
    if (resource.repr.family != artifact::Family::kGgml) {
      return Refused(std::format("{} is not a GGML representation", resource.name));
    }
    all.push_back({.roles = resource.roles,
                   .type = std::string(resource.repr.type),
                   .ne = resource.repr.dims});
  }
  const std::size_t first_array = all.size();
  for (const artifact::ExpertArray& array : artifact.expert_arrays()) {
    if (array.repr.family != artifact::Family::kGgml) {
      return Refused(std::format("{} is not a GGML representation", array.name));
    }
    all.push_back({.roles = {array.name},
                   .type = std::string(array.repr.type),
                   .ne = array.repr.dims,
                   .expert_array = true,
                   .count = array.count});
  }
  auto bound =
      BindDsv4Roles(profile, want_architecture, artifact.model().architecture, tables, all, extra);
  if (!bound) {
    return bound;
  }
  // Expert arrays are indexed among the artifact's expert arrays.
  const auto rebase = [&](Dsv4Tensor& t) { t.index -= static_cast<std::uint32_t>(first_array); };
  for (Dsv4Layer& l : bound->layers) {
    rebase(l.gate_exps);
    rebase(l.up_exps);
    rebase(l.down_exps);
  }
  return bound;
}

std::expected<void, std::string> CheckDsv4HashRouting(const Dsv4Profile& p,
                                                      std::span<const std::int32_t> table) {
  if (!ProfileIsSane(p) || table.size() != std::uint64_t{p.experts_used} * std::uint64_t{p.vocab}) {
    return Refused("a hash-routing table that is not [experts_used, vocab]");
  }
  for (std::size_t i = 0; i < table.size(); ++i) {
    if (table[i] < 0 || std::cmp_greater_equal(table[i], p.experts)) {
      return Refused(std::format("token {}'s hash route {} names expert {}, not one of {}",
                                 i / p.experts_used, i % p.experts_used, table[i], p.experts));
    }
  }
  return {};
}

// ---------------------------------------------------------------- state

std::int64_t Dsv4StateLayout::Find(std::uint32_t layer, Dsv4StateTensor::Kind kind) const {
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    if (tensors[i].layer == layer && tensors[i].kind == kind) {
      return static_cast<std::int64_t>(i);
    }
  }
  return -1;
}

std::vector<StateRepresentation> Dsv4StateLayout::Representations(std::uint32_t max_verify) const {
  using K = Dsv4StateTensor::Kind;
  std::uint64_t cells = 0;
  std::uint64_t caches = 0;
  std::uint64_t rings = 0;
  for (const Dsv4StateTensor& t : tensors) {
    switch (t.kind) {
      case K::kRawK:
        cells += t.bytes;
        break;
      case K::kCsaK:
      case K::kLidK:
      case K::kHcaK:
        caches += t.bytes;
        break;
      default:
        rings += t.bytes;
        break;
    }
  }
  const std::uint8_t capabilities = max_verify == 0
                                        ? static_cast<std::uint8_t>(StateCapability::kAppend)
                                        : (StateCapability::kAppend | StateCapability::kTruncate);
  const auto fixed = [capabilities](std::string name, std::uint64_t bytes) {
    return StateRepresentation{.name = std::move(name),
                               .block_positions = 0,
                               .block_bytes = Bytes(bytes),
                               .capabilities = capabilities,
                               .max_snapshots = 0,
                               .snapshot_bytes = Bytes(0)};
  };
  return {fixed("dsv4.window", cells), fixed("dsv4.compressed", caches),
          fixed("dsv4.compressor", rings)};
}

std::expected<Dsv4StateLayout, std::string> Dsv4State(const Dsv4Profile& p, std::uint32_t context,
                                                      std::uint32_t max_rows, Dsv4Window window) {
  if (!ProfileIsSane(p)) {
    return Refused("the profile is not a DeepSeek V4 model's");
  }
  // Positions are I32 in the graph, and the caches' cell counts (the
  // context padded to 256) must fit their 32-bit fields.
  if (context == 0 || context > kDsv4FlashContext || max_rows == 0 || max_rows > context ||
      context > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) - 255) {
    return Refused(std::format("no state for {} positions in chunks of {}", context, max_rows));
  }
  using K = Dsv4StateTensor::Kind;
  Dsv4StateLayout s;
  s.context = context;
  s.max_rows = max_rows;
  s.window = window;
  // The full-size window cache llama.cpp's contexts default to
  // (swa_full): a cell per position of the context; or a ring of the
  // window and a chunk, when that is smaller.
  s.raw_cells = static_cast<std::uint32_t>(Pad(context, 256));
  if (window == Dsv4Window::kRing) {
    s.raw_cells = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(s.raw_cells, Pad(std::uint64_t{p.window} + max_rows, 256)));
  }
  if (max_rows > s.raw_cells - p.window) {
    return Refused(std::format("chunks of {} rows leave the {}-position window no room in {} cells",
                               max_rows, p.window, s.raw_cells));
  }
  const auto comp_cells = [&](std::uint32_t ratio) {
    return static_cast<std::uint32_t>(
        Pad(std::max<std::uint64_t>(1, (std::uint64_t{context} + ratio - 1) / ratio), 256));
  };
  s.csa_cells = comp_cells(kDsv4CsaRatio);
  s.hca_cells = comp_cells(kDsv4HcaRatio);
  s.csa_state_rows = 2 * kDsv4CsaRatio;
  s.hca_state_rows = kDsv4HcaRatio;
  const auto add = [&](K kind, std::uint32_t layer, bool f16, std::uint64_t ne0,
                       std::uint64_t ne1) {
    const std::uint64_t bytes = ne0 * ne1 * (f16 ? 2 : 4);
    s.tensors.push_back({.kind = kind,
                         .layer = layer,
                         .f16 = f16,
                         .ne0 = ne0,
                         .ne1 = ne1,
                         .offset = s.bytes,
                         .bytes = bytes});
    s.bytes = Pad(s.bytes + bytes, 256);
  };
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    add(K::kRawK, il, true, p.head_dim, s.raw_cells);
    if (p.compress_ratios[il] == kDsv4CsaRatio) {
      add(K::kCsaK, il, true, p.head_dim, s.csa_cells);
      add(K::kCsaStateKv, il, false, 2ULL * p.head_dim, s.csa_state_rows);
      add(K::kCsaStateScore, il, false, 2ULL * p.head_dim, s.csa_state_rows);
      add(K::kLidK, il, true, p.indexer_head_dim, s.csa_cells);
      add(K::kLidStateKv, il, false, 2ULL * p.indexer_head_dim, s.csa_state_rows);
      add(K::kLidStateScore, il, false, 2ULL * p.indexer_head_dim, s.csa_state_rows);
    } else if (p.compress_ratios[il] == kDsv4HcaRatio) {
      add(K::kHcaK, il, true, p.head_dim, s.hca_cells);
      add(K::kHcaStateKv, il, false, p.head_dim, s.hca_state_rows);
      add(K::kHcaStateScore, il, false, p.head_dim, s.hca_state_rows);
    }
  }
  return s;
}

std::expected<std::vector<StateRange>, std::string> Dsv4UsedState(const Dsv4StateLayout& state,
                                                                  std::uint32_t positions) {
  if (positions > state.context) {
    return Refused("used state passes the DeepSeek context");
  }
  using K = Dsv4StateTensor::Kind;
  std::vector<StateRange> ranges;
  for (const Dsv4StateTensor& t : state.tensors) {
    std::uint64_t rows = t.ne1;
    bool compressed = false;
    if (t.kind == K::kRawK && state.window == Dsv4Window::kFull) {
      rows = std::min(t.ne1, Pad(positions, 256));
    } else if (t.kind == K::kCsaK || t.kind == K::kLidK || t.kind == K::kHcaK) {
      const std::uint64_t ratio = t.kind == K::kHcaK ? kDsv4HcaRatio : kDsv4CsaRatio;
      const std::uint64_t visible = (std::uint64_t{positions} + ratio - 1) / ratio;
      rows = std::min(t.ne1, std::max<std::uint64_t>(256, Pad(visible, 256)));
      compressed = true;
    }
    const std::uint64_t row_bytes = t.ne0 * (t.f16 ? 2 : 4);
    if (rows != 0) {
      ranges.push_back({.offset = t.offset, .bytes = rows * row_bytes});
    }
    if (compressed && rows < t.ne1) {
      ranges.push_back({.offset = t.offset + ((t.ne1 - 1) * row_bytes), .bytes = row_bytes});
    }
  }
  return ranges;
}

std::expected<std::vector<StateRange>, std::string> Dsv4CheckpointWrites(
    const Dsv4StateLayout& state, std::uint32_t positions) {
  if (positions > state.context) {
    return Refused("checkpoint passes the DeepSeek context");
  }
  using K = Dsv4StateTensor::Kind;
  std::vector<StateRange> writes;
  for (const Dsv4StateTensor& t : state.tensors) {
    std::uint64_t first = 0;
    if (t.kind == K::kRawK && state.window == Dsv4Window::kFull) {
      first = positions;
    } else if (t.kind == K::kCsaK || t.kind == K::kLidK || t.kind == K::kHcaK) {
      const std::uint64_t ratio = t.kind == K::kHcaK ? kDsv4HcaRatio : kDsv4CsaRatio;
      first = positions / ratio;
      // Every compressor chunk writes its final dummy cell, even outside
      // the visible prefix. Retained bytes must reproduce it exactly.
      const std::uint64_t row = t.ne0 * (t.f16 ? 2 : 4);
      writes.push_back({.offset = t.offset + ((t.ne1 - 1) * row), .bytes = row});
    }
    first = std::min(first, t.ne1);
    const std::uint64_t row = t.ne0 * (t.f16 ? 2 : 4);
    if (first < t.ne1) {
      writes.push_back({.offset = t.offset + (first * row), .bytes = (t.ne1 - first) * row});
    }
  }
  return writes;
}

std::uint32_t Dsv4MostRows(const Dsv4Profile& p, std::uint32_t context) {
  if (!ProfileIsSane(p) || context == 0 || context > kDsv4FlashContext ||
      context > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) - 255) {
    return 0;
  }
  const std::uint64_t cells = Pad(context, 256);
  if (cells <= p.window) {
    return 0;
  }
  // The fast plan's attention mask, F16 [ring cells + compressed cells,
  // rows], stays under 2^31 bytes: the MMA kernel takes its planes'
  // strides in 32 bits (RE-037). The largest such rows, by bisection (the
  // mask grows with the rows).
  const std::uint64_t compressed =
      Pad((std::uint64_t{context} + kDsv4CsaRatio - 1) / kDsv4CsaRatio, 256);
  const auto fits = [&](std::uint64_t rows) {
    const std::uint64_t ring = std::min(cells, Pad(std::uint64_t{p.window} + rows, 256));
    return (ring + compressed) * rows * 2 <=
           static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
  };
  std::uint64_t lo = 0;
  std::uint64_t hi = std::min<std::uint64_t>(context, cells - p.window);
  while (lo < hi) {
    const std::uint64_t mid = hi - ((hi - lo) / 2);
    if (fits(mid)) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return static_cast<std::uint32_t>(lo);
}

// ---------------------------------------------------------------- chunk inputs

std::expected<Dsv4CompGeometry, std::string> Dsv4CompressorGeometry(
    std::uint32_t ratio, bool overlap, std::uint32_t state_rows, std::uint32_t cache_rows,
    std::uint32_t n_past, std::uint32_t rows) {
  const std::uint64_t end = std::uint64_t{n_past} + rows;
  if (ratio == 0 || rows == 0 ||
      std::uint64_t{state_rows} < std::uint64_t{ratio} * (overlap ? 2U : 1U) || cache_rows == 0 ||
      end > std::numeric_limits<std::int32_t>::max())
    return Refused("not a bounded compressor geometry");
  const auto complete = end / ratio - n_past / ratio;
  const auto visible = end / ratio;
  const auto read = std::max<std::uint64_t>(Pad(visible, 256), 256);
  if ((complete != 0 && visible > cache_rows) || read > cache_rows ||
      (complete == 0 && ratio != kDsv4CsaRatio && visible >= cache_rows))
    return Refused("compressor geometry exceeds its cache");
  return Dsv4CompGeometry{.n_kv = static_cast<std::uint32_t>(read),
                          .blocks = static_cast<std::uint32_t>(
                              ratio == kDsv4CsaRatio ? (std::uint64_t{rows} + ratio - 1) / ratio
                                                     : std::max<std::uint64_t>(complete, 1)),
                          .persist = std::min(rows, state_rows)};
}

std::expected<Dsv4CompPlan, std::string> Dsv4CompressorPlan(std::uint32_t ratio, bool overlap,
                                                            std::uint32_t state_rows,
                                                            std::uint32_t cache_rows,
                                                            std::uint32_t n_past,
                                                            std::uint32_t rows) {
  auto geometry = Dsv4CompressorGeometry(ratio, overlap, state_rows, cache_rows, n_past, rows);
  if (!geometry) return Refused(geometry.error());
  const std::uint64_t end = std::uint64_t{n_past} + rows;
  if (end > std::numeric_limits<std::int32_t>::max()) {
    return Refused("positions beyond int32");
  }
  Dsv4CompPlan plan;
  plan.ratio = ratio;
  plan.overlap = overlap;
  plan.n_visible.resize(rows);
  // Rows of the graph's [ring state | chunk rows | zero row] source.
  const auto source = [&](std::int64_t pos) -> std::int32_t {
    if (pos < 0) {
      return static_cast<std::int32_t>(state_rows + rows);
    }
    if (std::cmp_greater_equal(pos, n_past) && std::cmp_less(pos, end)) {
      return static_cast<std::int32_t>(state_rows + (pos - n_past));
    }
    return static_cast<std::int32_t>(pos % state_rows);
  };
  std::map<std::int32_t, std::pair<std::int32_t, std::int64_t>> persist;  // dst -> (src, pos)
  std::vector<std::int32_t> prev_reads;
  std::vector<std::int32_t> cur_reads;
  std::int64_t n_kv = 0;
  for (std::uint32_t i = 0; i < rows; ++i) {
    const std::int64_t pos = std::int64_t{n_past} + i;
    plan.state_pos.push_back(static_cast<std::int32_t>(pos % ratio));
    const std::int64_t visible = (pos + 1) / ratio;
    plan.n_visible[i] = static_cast<std::int32_t>(visible);
    n_kv = std::max(n_kv, visible);
    const auto dst = static_cast<std::int32_t>(pos % state_rows);
    const auto it = persist.find(dst);
    if (it == persist.end() || pos > it->second.second) {
      persist[dst] = {static_cast<std::int32_t>(i), pos};
    }
    if ((pos + 1) % ratio != 0) {
      continue;
    }
    const std::int64_t start = pos + 1 - ratio;
    if (std::cmp_greater_equal(pos / ratio, cache_rows)) {
      return Refused(std::format("block {} is past the {}-row cache", pos / ratio, cache_rows));
    }
    plan.write_idxs.push_back(pos / ratio);
    plan.write_pos.push_back(static_cast<std::int32_t>(start));
    for (std::uint32_t j = 0; j < ratio; ++j) {
      if (overlap) {
        prev_reads.push_back(source(start - ratio + j));
        cur_reads.push_back(source(start + j));
      } else {
        plan.read_idxs.push_back(source(start + j));
      }
    }
  }
  // llama.cpp keeps the compressor's operations in every chunk's graph: CSA
  // (and the indexer) pad to a block per ratio rows, HCA to one block, each
  // missing block a dummy written to the cache's last row, which stays
  // masked (dsv4_build_comp_plan).
  const auto dummy = [&] {
    plan.write_idxs.push_back(std::int64_t{cache_rows} - 1);
    plan.write_pos.push_back(0);
    const std::int32_t src = source(n_past);
    for (std::uint32_t j = 0; j < ratio; ++j) {
      if (overlap) {
        prev_reads.push_back(src);
        cur_reads.push_back(src);
      } else {
        plan.read_idxs.push_back(src);
      }
    }
  };
  if (ratio == kDsv4CsaRatio) {
    const std::uint32_t blocks = (rows + ratio - 1) / ratio;
    if (plan.blocks() < blocks) {
      if (plan.blocks() + 1 != blocks) {
        return Refused("the chunk's positions complete too few blocks");
      }
      dummy();
    }
  } else if (plan.write_idxs.empty()) {
    if (std::cmp_greater_equal(n_kv, cache_rows)) {
      return Refused("the dummy block would overwrite a visible row");
    }
    dummy();
  }
  if (overlap) {
    plan.read_idxs = std::move(prev_reads);
    plan.read_idxs.insert(plan.read_idxs.end(), cur_reads.begin(), cur_reads.end());
  }
  plan.n_kv = static_cast<std::uint32_t>(
      std::max<std::uint64_t>(Pad(static_cast<std::uint64_t>(n_kv), 256), 256));
  if (plan.n_kv > cache_rows) {
    return Refused("the visible rows exceed the cache");
  }
  for (const auto& [dst, entry] : persist) {
    plan.persist_src.push_back(entry.first);
    plan.persist_dst.push_back(dst);
  }
  if (plan.n_kv != geometry->n_kv || plan.blocks() != geometry->blocks ||
      plan.persist_src.size() != geometry->persist)
    return Refused("compressor inputs disagree with their geometry");
  return plan;
}

std::expected<Dsv4ChunkInputs, std::string> Dsv4Chunk(const Dsv4Profile& profile,
                                                      const Dsv4StateLayout& state,
                                                      std::uint32_t n_past, std::uint32_t rows,
                                                      bool masks, bool raw_mask) {
  if (rows == 0 || rows > state.max_rows || n_past > state.context ||
      rows > state.context - n_past) {
    return Refused(std::format("{} rows after {} do not fit a {}-position state of {}-row chunks",
                               rows, n_past, state.context, state.max_rows));
  }
  Dsv4ChunkInputs in;
  in.n_past = n_past;
  in.rows = rows;
  const std::uint64_t total = std::uint64_t{n_past} + rows;
  const std::uint64_t cells = state.raw_cells;
  const bool ring = state.window == Dsv4Window::kRing;
  in.raw_n_kv = ring ? state.raw_cells
                     : static_cast<std::uint32_t>(std::min<std::uint64_t>(
                           cells, std::max<std::uint64_t>(256, Pad(std::min(total, cells), 256))));
  in.positions.resize(rows);
  in.raw_cells.resize(rows);
  if (raw_mask) in.raw_mask.assign(std::size_t{rows} * in.raw_n_kv, kHalfNegInf);
  for (std::uint32_t i = 0; i < rows; ++i) {
    const std::uint64_t pos = n_past + i;
    in.positions[i] = static_cast<std::int32_t>(pos);
    in.raw_cells[i] = static_cast<std::int64_t>(pos % cells);
    if (raw_mask) {
      const auto visible = HostMaskRing(in.raw_n_kv, pos, total, cells, profile.window);
      if (!visible) return Refused("invalid raw causal-mask interval");
      FillHostMaskVisible(std::span{in.raw_mask}.subspan(std::size_t{i} * in.raw_n_kv, in.raw_n_kv),
                          *visible, kHalfZero);
    }
  }
  auto csa =
      Dsv4CompressorPlan(kDsv4CsaRatio, true, state.csa_state_rows, state.csa_cells, n_past, rows);
  auto hca =
      Dsv4CompressorPlan(kDsv4HcaRatio, false, state.hca_state_rows, state.hca_cells, n_past, rows);
  if (!csa || !hca) {
    return Refused(!csa ? csa.error() : hca.error());
  }
  in.csa = std::move(*csa);
  in.lid = in.csa;  // the indexer compresses with CSA's ratio, overlap and ring
  in.hca = std::move(*hca);
  const auto mask =
      [&](const Dsv4CompPlan& plan) -> std::expected<std::vector<std::uint16_t>, std::string> {
    std::vector<std::uint16_t> m(std::size_t{rows} * plan.n_kv, kHalfNegInf);
    if (plan.n_visible.size() != rows) return Refused("invalid compressor mask rows");
    for (std::uint32_t i = 0; i < rows; ++i) {
      if (plan.n_visible[i] < 0) return Refused("negative compressor visibility");
      const auto visible = HostMaskPrefix(plan.n_kv, static_cast<std::uint64_t>(plan.n_visible[i]));
      if (!visible) return Refused("compressor visibility exceeds mask width");
      FillHostMaskVisible(std::span{m}.subspan(std::size_t{i} * plan.n_kv, plan.n_kv), *visible,
                          kHalfZero);
    }
    return m;
  };
  if (masks) {
    auto csa_mask = mask(in.csa);
    auto hca_mask = mask(in.hca);
    auto lid_mask = mask(in.lid);
    if (!csa_mask || !hca_mask || !lid_mask)
      return Refused(!csa_mask   ? csa_mask.error()
                     : !hca_mask ? hca_mask.error()
                                 : lid_mask.error());
    in.csa_mask = std::move(*csa_mask);
    in.hca_mask = std::move(*hca_mask);
    in.lid_mask = std::move(*lid_mask);
  }
  return in;
}

// ---------------------------------------------------------------- speculation

namespace {

// The widths Dsv4Chunk gives a chunk ending before `total`: the window
// cells attention reads and each compressor's mask.
struct Widths {
  std::uint64_t raw = 0;
  std::uint64_t csa = 0;
  std::uint64_t hca = 0;
  bool operator==(const Widths&) const = default;
};

Widths WidthsAt(const Dsv4StateLayout& state, std::uint64_t total) {
  const std::uint64_t cells = state.raw_cells;
  const auto comp = [&](std::uint32_t ratio) {
    return std::max<std::uint64_t>(Pad(total / ratio, 256), 256);
  };
  return {.raw = state.window == Dsv4Window::kRing
                     ? cells
                     : std::min<std::uint64_t>(
                           cells, std::max<std::uint64_t>(256, Pad(std::min(total, cells), 256))),
          .csa = comp(kDsv4CsaRatio),
          .hca = comp(kDsv4HcaRatio)};
}

}  // namespace

bool Dsv4SameWidths(const Dsv4StateLayout& state, std::uint32_t n_past, std::uint32_t rows) {
  if (rows == 0) {
    return false;
  }
  // Each width grows with the position: equal at the first and last rows,
  // equal at every row between.
  return WidthsAt(state, std::uint64_t{n_past} + 1) ==
         WidthsAt(state, std::uint64_t{n_past} + rows);
}

Dsv4Writes Dsv4ChunkWrites(const Dsv4Profile& profile, const Dsv4StateLayout& state,
                           const Dsv4ChunkInputs& chunk) {
  using K = Dsv4StateTensor::Kind;
  Dsv4Writes out;
  out.rows.resize(chunk.rows);
  const auto tensor = [&](std::uint32_t layer, K kind) -> const Dsv4StateTensor* {
    const std::int64_t i = state.Find(layer, kind);
    return i < 0 ? nullptr : &state.tensors[static_cast<std::size_t>(i)];
  };
  const auto row_bytes = [](const Dsv4StateTensor& t) { return t.ne0 * (t.f16 ? 2U : 4U); };
  const auto at = [&](const Dsv4StateTensor& t, std::uint64_t row) {
    return StateRange{.offset = t.offset + (row * row_bytes(t)), .bytes = row_bytes(t)};
  };
  // A compressor's ring rows (by the chunk row that persists each) and its
  // blocks' compressed rows (by the row completing each; the dummy's are
  // scratch).
  const auto compressor = [&](const Dsv4CompPlan& plan, const Dsv4StateTensor* kv,
                              const Dsv4StateTensor* score, const Dsv4StateTensor* cache) {
    for (std::size_t i = 0; i < plan.persist_src.size(); ++i) {
      const auto src = static_cast<std::size_t>(plan.persist_src[i]);
      const auto dst = static_cast<std::uint64_t>(plan.persist_dst[i]);
      out.rows.at(src).push_back(at(*kv, dst));
      out.rows.at(src).push_back(at(*score, dst));
    }
    for (std::size_t b = 0; b < plan.write_idxs.size(); ++b) {
      const auto block = static_cast<std::uint64_t>(plan.write_idxs[b]);
      const std::uint64_t last = (block * plan.ratio) + plan.ratio - 1;
      const bool real = std::cmp_equal(plan.write_pos[b], block * plan.ratio) &&
                        last >= chunk.n_past && last < std::uint64_t{chunk.n_past} + chunk.rows;
      if (real) {
        out.rows.at(last - chunk.n_past).push_back(at(*cache, block));
      } else {
        out.scratch.push_back(at(*cache, block));
      }
    }
  };
  for (std::uint32_t il = 0; il < profile.layers; ++il) {
    const Dsv4StateTensor* raw = tensor(il, K::kRawK);
    for (std::uint32_t i = 0; i < chunk.rows; ++i) {
      out.rows[i].push_back(at(*raw, static_cast<std::uint64_t>(chunk.raw_cells[i])));
    }
    const std::uint32_t ratio = profile.compress_ratios[il];
    if (ratio == kDsv4CsaRatio) {
      compressor(chunk.csa, tensor(il, K::kCsaStateKv), tensor(il, K::kCsaStateScore),
                 tensor(il, K::kCsaK));
      compressor(chunk.lid, tensor(il, K::kLidStateKv), tensor(il, K::kLidStateScore),
                 tensor(il, K::kLidK));
    } else if (ratio == kDsv4HcaRatio) {
      compressor(chunk.hca, tensor(il, K::kHcaStateKv), tensor(il, K::kHcaStateScore),
                 tensor(il, K::kHcaK));
    }
  }
  return out;
}

std::uint64_t Dsv4VerifySnapshotBytes(const Dsv4Profile& profile, const Dsv4StateLayout& state,
                                      std::uint32_t max_rows) {
  using K = Dsv4StateTensor::Kind;
  // At most: every row writes its cell, its ring rows and a compressed row
  // in every layer, and every compressed cache's scratch row is written.
  std::uint64_t per_row = 0;
  std::uint64_t scratch = 0;
  for (const Dsv4StateTensor& t : state.tensors) {
    const std::uint64_t row = t.ne0 * (t.f16 ? 2U : 4U);
    per_row += row;
    if (t.kind == K::kCsaK || t.kind == K::kLidK || t.kind == K::kHcaK) {
      scratch += row;
    }
  }
  (void)profile;
  return (per_row * max_rows) + scratch;
}

}  // namespace llmp::model
