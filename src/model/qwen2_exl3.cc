// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "model/qwen2_exl3.h"

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
#include <tuple>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "artifact/representation.h"
#include "base/sha256.h"

namespace llmp::model {
namespace {

std::unexpected<std::string> Refused(std::string detail) {
  return std::unexpected(std::move(detail));
}

// Upstream's path thresholds (the header).
constexpr int kPackedMaxRows = 144;
constexpr int kFusedMinRows = 1024;
constexpr int kMultiMaxRows = 32;
constexpr int kGemvMaxRows = 8;
constexpr int kKqStride = 256;
constexpr std::uint64_t kAlignment = 256;

std::uint64_t RoundUp(std::uint64_t bytes, std::uint64_t to) { return (bytes + to - 1) / to * to; }

// ------------------------------------------------------------------ binding

class Binder {
 public:
  explicit Binder(std::span<const Exl3Resource> resources) : resources_(resources) {
    for (std::uint32_t i = 0; i < resources_.size(); ++i) {
      by_name_.emplace(resources_[i].name, i);
    }
  }

  std::expected<std::uint32_t, std::string> Plain(const std::string& name, std::string_view dtype,
                                                  const std::vector<std::uint64_t>& shape) {
    auto index = Find(name);
    if (!index) {
      return std::unexpected(index.error());
    }
    const Exl3Resource& r = resources_[*index];
    if (r.family != "plain" || r.dtype != dtype || r.shape != shape) {
      return Refused(std::format("{} is not plain {} of the profile's shape", name, dtype));
    }
    return *index;
  }

  std::expected<Exl3LinearBinding, std::string> Linear(std::string name, int k, int n, bool bias) {
    Exl3LinearBinding linear;
    linear.name = std::move(name);
    linear.k = k;
    linear.n = n;
    auto trellis = Find(linear.name + ".trellis");
    if (!trellis) {
      return std::unexpected(trellis.error());
    }
    const Exl3Resource& t = resources_[*trellis];
    const auto bits = static_cast<int>(t.k_bits);
    if (t.family != "exl3" || t.role != "trellis" || t.codebook != "mcg" || t.dtype != "I16" ||
        std::cmp_not_equal(t.in_features, k) || std::cmp_not_equal(t.out_features, n) ||
        (bits != 4 && bits != 5 && bits != 6 && bits != 8) ||
        t.shape != std::vector<std::uint64_t>{static_cast<std::uint64_t>(k / 16),
                                              static_cast<std::uint64_t>(n / 16),
                                              static_cast<std::uint64_t>(16 * bits)}) {
      return Refused(std::format("{}.trellis is not an mcg trellis of {} × {} at K = 4, 5, 6 or 8",
                                 linear.name, k, n));
    }
    linear.trellis = *trellis;
    linear.bits = bits;
    for (const auto& [role, length, out] :
         {std::tuple{"suh", k, &linear.suh}, std::tuple{"svh", n, &linear.svh}}) {
      auto index = Find(std::format("{}.{}", linear.name, role));
      if (!index) {
        return std::unexpected(index.error());
      }
      const Exl3Resource& r = resources_[*index];
      if (r.family != "exl3" || r.role != role || r.dtype != "F16" ||
          r.shape != std::vector<std::uint64_t>{static_cast<std::uint64_t>(length)}) {
        return Refused(
            std::format("{}.{} is not an F16 side vector of {}", linear.name, role, length));
      }
      *out = *index;
    }
    auto mcg = Find(linear.name + ".mcg");
    if (!mcg) {
      return std::unexpected(mcg.error());
    }
    const Exl3Resource& m = resources_[*mcg];
    if (m.family != "exl3" || m.role != "mcg" || m.dtype != "I32" || !m.shape.empty()) {
      return Refused(linear.name + ".mcg is not the mcg flag");
    }
    if (bias) {
      auto b = Plain(linear.name + ".bias", "F16", {static_cast<std::uint64_t>(n)});
      if (!b) {
        return std::unexpected(b.error());
      }
      linear.bias = *b;
    }
    return linear;
  }

  // Every resource was bound.
  std::expected<void, std::string> AllBound() const {
    for (std::uint32_t i = 0; i < resources_.size(); ++i) {
      if (!bound_.contains(i)) {
        return Refused(
            std::format("the artifact holds {}, which the plan does not read", resources_[i].name));
      }
    }
    return {};
  }

 private:
  std::expected<std::uint32_t, std::string> Find(const std::string& name) {
    const auto found = by_name_.find(name);
    if (found == by_name_.end()) {
      return Refused(std::format("the artifact has no {}", name));
    }
    bound_.insert({found->second, true});
    return found->second;
  }

  std::span<const Exl3Resource> resources_;
  std::map<std::string, std::uint32_t, std::less<>> by_name_;
  std::map<std::uint32_t, bool> bound_;
};

// ------------------------------------------------------------------ planning

void Field(base::Sha256& hash, std::string_view text) {
  const std::uint64_t length = text.size();
  hash.Update(std::as_bytes(std::span(&length, 1)));
  hash.Update(text);
}

std::string Describe(const Exl3LinearPlan& plan) {
  std::string out =
      std::format("{} shape {} blocks {} concurrency {} config {}", Exl3PathName(plan.path),
                  plan.shape, plan.blocks, plan.concurrency, plan.config);
  for (const Exl3Pin& pin : plan.pins) {
    out += std::format(" | {} m {} k {} n {} ldc {}", pin.f32 ? "HSS" : "HSH", pin.m, pin.k, pin.n,
                       pin.ldc);
    for (const std::uint64_t value : pin.config) {
      out += std::format(" {}", value);
    }
  }
  return out;
}

// The one path upstream takes for a linear at `rows`, and whether the
// table's plan may serve it.
std::expected<void, std::string> CheckPath(const Exl3LinearPlan& plan, int rows, Exl3Arm arm,
                                           bool multi, const Exl3LinearBinding& linear, bool f32) {
  if (multi) {
    if (plan.path != Exl3Path::kMulti || plan.shape <= 0 || plan.blocks <= 0 ||
        plan.concurrency <= 0) {
      return Refused("gate and up through the multi-GEMM, with its shape, grid and concurrency");
    }
    return {};
  }
  if (rows <= kPackedMaxRows) {
    const bool gemm = plan.path == Exl3Path::kGemm && plan.shape > 0 && plan.blocks > 0;
    const bool gemv = plan.path == Exl3Path::kGemv && arm == Exl3Arm::kO && rows <= kGemvMaxRows &&
                      plan.blocks > 0 && (plan.config == 0 || plan.config == 1);
    if (!gemm && !gemv) {
      return Refused(std::format("a packed plan at {} rows ({}), GEMV only in EXL3-O to 8 rows",
                                 rows, Exl3PathName(plan.path)));
    }
    return {};
  }
  const Exl3Path path =
      rows >= kFusedMinRows ? Exl3Path::kReconstructFused : Exl3Path::kReconstruct;
  if (plan.path != path) {
    return Refused(std::format("upstream reconstructs at {} rows ({}), not {}", rows,
                               Exl3PathName(path), Exl3PathName(plan.path)));
  }
  const std::vector<int> slices = Exl3Slices(linear.n);
  if (plan.pins.size() != slices.size()) {
    return Refused(
        std::format("{} slices need {} pins, not {}", linear.n, slices.size(), plan.pins.size()));
  }
  for (std::size_t s = 0; s < slices.size(); ++s) {
    const Exl3Pin& pin = plan.pins[s];
    if (pin.m != rows || pin.k != linear.k || pin.n != slices[s] || pin.ldc != linear.n ||
        pin.f32 != f32) {
      return Refused(std::format("slice {}'s pin was recorded for another GEMM", s));
    }
  }
  return {};
}

struct Buffer {
  std::vector<std::string> names;  // the tensors it holds (planes of gate_up)
  std::uint64_t bytes = 0;
  int first = std::numeric_limits<int>::max();
  int last = -1;
};

}  // namespace

const Qwen2Profile& Qwen25Instruct05BExl3() {
  // config.json of both EXL3 fixtures (docs/exl3-bringup.md): the same
  // model as the FP16 fixture; max_position_embeddings 32,768 is RoPE's
  // original context (exl3-op-plan.json, rope.n_ctx_orig).
  static constexpr Qwen2Profile kProfile = {.name = "qwen2.5-0.5b-instruct-exl3",
                                            .layers = 24,
                                            .width = 896,
                                            .heads = 14,
                                            .kv_heads = 2,
                                            .head_dim = 64,
                                            .ffn = 4864,
                                            .vocab = 151936,
                                            .train_context = 32768,
                                            .rms_eps = 1e-6f,
                                            .rope_base = 1000000.0f,
                                            .weight_type = "exl3"};
  return kProfile;
}

std::expected<Exl3Binding, std::string> BindQwen2Exl3(const Qwen2Profile& profile,
                                                      const artifact::Artifact& artifact) {
  if (!artifact.expert_arrays().empty()) {
    return Refused("a dense Qwen2 artifact has no expert arrays");
  }
  std::vector<Exl3Resource> resources;
  resources.reserve(artifact.resources().size());
  for (const artifact::Resource& r : artifact.resources()) {
    if (r.roles.size() != 1 || r.roles.front() != r.name) {
      return Refused(std::format("{} has aliases, which the EXL3 plan does not bind", r.name));
    }
    resources.push_back({.name = r.name,
                         .family = std::string(artifact::FamilyName(r.repr.family)),
                         .dtype = std::string(r.repr.type),
                         .shape = r.repr.dims,
                         .role = std::string(r.repr.role),
                         .k_bits = r.repr.k_bits,
                         .in_features = r.repr.in_features,
                         .out_features = r.repr.out_features,
                         .codebook = std::string(r.repr.codebook)});
  }
  return BindQwen2Exl3(profile, artifact.model().architecture, resources);
}

std::expected<Exl3Binding, std::string> BindQwen2Exl3(const Qwen2Profile& p,
                                                      std::string_view architecture,
                                                      std::span<const Exl3Resource> resources) {
  if (architecture != "qwen2") {
    return Refused(std::format("the artifact's architecture is {}, not qwen2", architecture));
  }
  // Every count nonzero before any is divided by or taken as a shape.
  if (p.layers == 0 || p.width == 0 || p.heads == 0 || p.kv_heads == 0 || p.head_dim == 0 ||
      p.ffn == 0 || p.vocab == 0 || p.width % 128 != 0 || p.ffn % 128 != 0 ||
      p.heads % p.kv_heads != 0 || (std::uint64_t{p.heads} * p.head_dim) != p.width ||
      p.kv_width() % 128 != 0) {
    return Refused("a profile the EXL3 plan cannot run");
  }
  Binder bind(resources);
  Exl3Binding out;
  const auto width = static_cast<int>(p.width);
  const auto ffn = static_cast<int>(p.ffn);
  const auto kv = static_cast<int>(p.kv_width());
  auto embed = bind.Plain("model.embed_tokens.weight", "BF16", {p.vocab, p.width});
  auto norm = bind.Plain("model.norm.weight", "BF16", {p.width});
  auto head = bind.Linear("lm_head", width, static_cast<int>(p.vocab), false);
  if (!embed) {
    return std::unexpected(embed.error());
  }
  if (!norm) {
    return std::unexpected(norm.error());
  }
  if (!head) {
    return std::unexpected(head.error());
  }
  out.embed = *embed;
  out.final_norm = *norm;
  out.lm_head = std::move(*head);
  for (std::uint32_t l = 0; l < p.layers; ++l) {
    const std::string b = std::format("model.layers.{}.", l);
    Exl3LayerBinding layer;
    auto attn = bind.Plain(b + "input_layernorm.weight", "BF16", {p.width});
    auto mlp = bind.Plain(b + "post_attention_layernorm.weight", "BF16", {p.width});
    if (!attn || !mlp) {
      return std::unexpected(!attn ? attn.error() : mlp.error());
    }
    layer.attn_norm = *attn;
    layer.mlp_norm = *mlp;
    struct Want {
      Exl3LinearBinding* into;
      std::string name;
      int k, n;
      bool bias;
    };
    for (const Want& w : {Want{&layer.q, b + "self_attn.q_proj", width, width, true},
                          Want{&layer.k, b + "self_attn.k_proj", width, kv, true},
                          Want{&layer.v, b + "self_attn.v_proj", width, kv, true},
                          Want{&layer.o, b + "self_attn.o_proj", width, width, false},
                          Want{&layer.gate, b + "mlp.gate_proj", width, ffn, false},
                          Want{&layer.up, b + "mlp.up_proj", width, ffn, false},
                          Want{&layer.down, b + "mlp.down_proj", ffn, width, false}}) {
      auto linear = bind.Linear(w.name, w.k, w.n, w.bias);
      if (!linear) {
        return std::unexpected(linear.error());
      }
      *w.into = std::move(*linear);
    }
    out.layers.push_back(std::move(layer));
  }
  if (auto all = bind.AllBound(); !all) {
    return std::unexpected(all.error());
  }
  return out;
}

std::string_view Exl3PathName(Exl3Path path) {
  switch (path) {
    case Exl3Path::kGemm:
      return "gemm";
    case Exl3Path::kGemv:
      return "gemv";
    case Exl3Path::kMulti:
      return "multi";
    case Exl3Path::kReconstruct:
      return "recon";
    case Exl3Path::kReconstructFused:
      return "fused";
  }
  return "unknown";
}

std::expected<void, std::string> Exl3LaunchTable::Add(std::string linear, int rows,
                                                      Exl3LinearPlan plan, std::string second) {
  auto key = std::make_pair(std::move(linear), rows);
  if (plans_.contains(key)) {
    return Refused(std::format("two plans for {} at {} rows", key.first, rows));
  }
  plans_.emplace(std::move(key), std::make_pair(std::move(plan), std::move(second)));
  return {};
}

const Exl3LinearPlan* Exl3LaunchTable::Find(std::string_view linear, int rows) const {
  const auto found = plans_.find(std::make_pair(std::string(linear), rows));
  return found == plans_.end() ? nullptr : &found->second.first;
}

std::string_view Exl3LaunchTable::Second(std::string_view linear, int rows) const {
  const auto found = plans_.find(std::make_pair(std::string(linear), rows));
  return found == plans_.end() ? std::string_view() : std::string_view(found->second.second);
}

int PaddedCells(const Exl3Phase& phase) {
  const int attended = phase.past + phase.rows;
  return (attended + kKqStride - 1) / kKqStride * kKqStride;
}

bool RecordedPhase(const Exl3Phase& phase) {
  if (phase.rows == 1 && phase.past > 0) {
    const int padded = PaddedCells(phase);
    return padded == 256 || padded == 1024 || padded == 1280;
  }
  return phase.past == 0 && (phase.rows == 32 || phase.rows == 144 || phase.rows == 145 ||
                             phase.rows == 1023 || phase.rows == 1024);
}

std::string_view Exl3DtypeName(Exl3Dtype dtype) {
  switch (dtype) {
    case Exl3Dtype::kF32:
      return "float32";
    case Exl3Dtype::kF16:
      return "float16";
    case Exl3Dtype::kBF16:
      return "bfloat16";
    case Exl3Dtype::kI32:
      return "int32";
  }
  return "unknown";
}

std::uint64_t Exl3DtypeBytes(Exl3Dtype dtype) {
  return dtype == Exl3Dtype::kF16 || dtype == Exl3Dtype::kBF16 ? 2 : 4;
}

std::uint64_t Exl3Tensor::bytes() const {
  std::uint64_t count = 1;
  for (const std::int64_t extent : shape) {
    count *= static_cast<std::uint64_t>(extent);
  }
  return count * Exl3DtypeBytes(dtype);
}

std::vector<int> Exl3Slices(int n) {
  std::vector<int> slices;
  for (int at = 0; at < n; at += kExl3SliceColumns) {
    slices.push_back(std::min(kExl3SliceColumns, n - at));
  }
  return slices;
}

std::expected<Exl3PhasePlan, std::string> PlanPhase(const Qwen2Profile& p,
                                                    const Exl3Binding& binding,
                                                    const Exl3LaunchTable& table, Exl3Arm arm,
                                                    const Exl3Phase& phase) {
  if (!RecordedPhase(phase)) {
    return Refused(std::format(
        "no recorded phase kind holds {} rows at position {} (padded K {}): record it first",
        phase.rows, phase.past, PaddedCells(phase)));
  }
  if (binding.layers.size() != p.layers) {
    return Refused("a binding of another profile");
  }
  Exl3PhasePlan plan;
  plan.phase = phase;
  plan.padded = PaddedCells(phase);
  plan.arm = arm;
  const std::int64_t n = phase.rows;
  const std::int64_t width = p.width;
  const std::int64_t heads = p.heads;
  const std::int64_t kv_heads = p.kv_heads;
  const std::int64_t head = p.head_dim;
  const std::int64_t kv = p.kv_width();
  const std::int64_t ffn = p.ffn;
  using D = Exl3Dtype;
  const auto tensor = [&](std::string name, D dtype, std::vector<std::int64_t> shape) {
    plan.tensors.emplace(name, Exl3Tensor{.name = name, .dtype = dtype, .shape = std::move(shape)});
  };
  tensor("ids", D::kI32, {n});
  tensor("positions", D::kI32, {n});
  tensor("mask", D::kF16, {n, plan.padded});
  tensor("embed.out", D::kF32, {n, width});
  tensor("resid.in", D::kF32, {n, width});
  tensor("attn_norm.f32", D::kF32, {n, width});
  tensor("attn_norm.out", D::kF16, {n, width});
  tensor("q_proj.gemm", D::kF16, {n, width});
  tensor("q", D::kF16, {n, width});
  tensor("k_proj.gemm", D::kF16, {n, kv});
  tensor("k", D::kF16, {n, kv});
  tensor("v_proj.gemm", D::kF16, {n, kv});
  tensor("v", D::kF16, {n, kv});
  tensor("rope_q.in", D::kF32, {n, heads, head});
  tensor("q_rope", D::kF32, {n, heads, head});
  tensor("rope_k.in", D::kF32, {n, kv_heads, head});
  tensor("k_rope.f32", D::kF32, {n, kv_heads, head});
  tensor("k_rope", D::kF16, {n, kv_heads, head});
  tensor("attn.f32", D::kF32, {n, heads, head});
  tensor("attn.out", D::kF16, {n, width});
  tensor("o_proj.out", D::kF32, {n, width});
  tensor("resid.mid", D::kF32, {n, width});
  tensor("mlp_norm.f32", D::kF32, {n, width});
  tensor("mlp_norm.out", D::kF16, {n, width});
  tensor("gate", D::kF32, {n, ffn});
  tensor("up", D::kF32, {n, ffn});
  tensor("swiglu.f32", D::kF32, {n, ffn});
  tensor("swiglu.out", D::kF16, {n, ffn});
  tensor("down_proj.out", D::kF32, {n, width});
  tensor("resid.out", D::kF32, {n, width});
  tensor("final_norm.f32", D::kF32, {n, width});
  tensor("final_norm.out", D::kF16, {n, width});
  tensor("logits", D::kF16, {n, static_cast<std::int64_t>(p.vocab)});

  const auto op = [](std::string name, int layer, Exl3Owner owner, std::string implementation,
                     std::vector<std::string> inputs, std::vector<std::string> outputs) {
    Exl3Op out;
    out.name = std::move(name);
    out.layer = layer;
    out.owner = owner;
    out.implementation = std::move(implementation);
    out.inputs = std::move(inputs);
    out.outputs = std::move(outputs);
    return out;
  };
  const std::string convert = "ggml.convert";
  const std::string norm = "ggml.rms_norm_mul.fused";
  // The linear ops, each with its plan from the table.
  const auto linear = [&](std::string name, int layer, const Exl3LinearBinding& l,
                          std::string input, std::string output,
                          bool f32) -> std::expected<Exl3Op, std::string> {
    const Exl3LinearPlan* launch = table.Find(l.name, phase.rows);
    if (launch == nullptr) {
      return Refused(
          std::format("the launch table has no plan for {} at {} rows", l.name, phase.rows));
    }
    if (auto checked = CheckPath(*launch, phase.rows, arm, false, l, f32); !checked) {
      return Refused(std::format("{} at {} rows: {}", l.name, phase.rows, checked.error()));
    }
    std::string implementation;
    switch (launch->path) {
      case Exl3Path::kGemm:
        implementation = "exl3.linear.gemm";
        break;
      case Exl3Path::kGemv:
        implementation = "exl3.linear.gemv";
        break;
      case Exl3Path::kReconstruct:
        implementation = "exl3.linear.reconstruct";
        break;
      case Exl3Path::kReconstructFused:
        implementation = "exl3.linear.reconstruct_fused";
        break;
      case Exl3Path::kMulti:
        break;
    }
    Exl3Op out = op(std::move(name), layer, Exl3Owner::kExl3, std::move(implementation),
                    {std::move(input)}, {std::move(output)});
    out.linear = l.name;
    out.launch = *launch;
    return out;
  };

  plan.ops.push_back(op("inputs", -1, Exl3Owner::kHost, "", {}, {"ids", "positions", "mask"}));
  plan.ops.push_back(
      op("embed", -1, Exl3Owner::kGgml, "ggml.get_rows", {"embed_table", "ids"}, {"embed.out"}));
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    const auto l = static_cast<int>(il);
    const Exl3LayerBinding& b = binding.layers[il];
    std::vector<Exl3Op> ops;
    ops.push_back(
        op("attn_norm", l, Exl3Owner::kGgml, norm, {"resid.in", "attn_norm.w"}, {"attn_norm.f32"}));
    ops.push_back(
        op("attn_norm.cast", l, Exl3Owner::kGgml, convert, {"attn_norm.f32"}, {"attn_norm.out"}));
    for (const auto& [proj, lin, out] :
         {std::tuple{"q_proj", &b.q, "q"}, std::tuple{"k_proj", &b.k, "k"},
          std::tuple{"v_proj", &b.v, "v"}}) {
      const std::string name = proj;
      auto made = linear(name, l, *lin, "attn_norm.out", name + ".gemm", false);
      if (!made) {
        return std::unexpected(made.error());
      }
      ops.push_back(std::move(*made));
      ops.push_back(op(name + ".bias_add", l, Exl3Owner::kExl3, "exl3.bias_add",
                       {name + ".gemm", name + ".bias"}, {out}));
    }
    ops.push_back(op("rope_q.cast", l, Exl3Owner::kGgml, convert, {"q"}, {"rope_q.in"}));
    ops.push_back(op("rope_q", l, Exl3Owner::kGgml, "ggml.rope.neox", {"rope_q.in", "positions"},
                     {"q_rope"}));
    ops.push_back(op("rope_k.cast", l, Exl3Owner::kGgml, convert, {"k"}, {"rope_k.in"}));
    ops.push_back(op("rope_k", l, Exl3Owner::kGgml, "ggml.rope.neox", {"rope_k.in", "positions"},
                     {"k_rope.f32"}));
    ops.push_back(op("rope_k.cast_out", l, Exl3Owner::kGgml, convert, {"k_rope.f32"}, {"k_rope"}));
    ops.push_back(op("kv_write.k", l, Exl3Owner::kCopy, "", {"k_rope"}, {"k_cache"}));
    ops.push_back(op("kv_write.v", l, Exl3Owner::kCopy, "", {"v"}, {"v_cache"}));
    ops.push_back(op("attention", l, Exl3Owner::kGgml, "ggml.flash_attn_ext.vec",
                     {"q_rope", "k_cache", "v_cache", "mask"}, {"attn.f32"}));
    ops.push_back(op("attention.cast", l, Exl3Owner::kGgml, convert, {"attn.f32"}, {"attn.out"}));
    {
      auto made = linear("o_proj", l, b.o, "attn.out", "o_proj.out", true);
      if (!made) {
        return std::unexpected(made.error());
      }
      ops.push_back(std::move(*made));
    }
    ops.push_back(op("attn_residual_add", l, Exl3Owner::kGgml, "ggml.add",
                     {"resid.in", "o_proj.out"}, {"resid.mid"}));
    ops.push_back(
        op("mlp_norm", l, Exl3Owner::kGgml, norm, {"resid.mid", "mlp_norm.w"}, {"mlp_norm.f32"}));
    ops.push_back(
        op("mlp_norm.cast", l, Exl3Owner::kGgml, convert, {"mlp_norm.f32"}, {"mlp_norm.out"}));
    if (phase.rows <= kMultiMaxRows) {
      const Exl3LinearPlan* launch = table.Find(b.gate.name, phase.rows);
      if (launch == nullptr || table.Second(b.gate.name, phase.rows) != b.up.name) {
        return Refused(
            std::format("the launch table has no multi-GEMM plan for {} and {} at {} "
                        "rows",
                        b.gate.name, b.up.name, phase.rows));
      }
      if (b.gate.bits != b.up.bits) {
        return Refused(std::format("{}: gate and up at different rates", b.gate.name));
      }
      if (auto checked = CheckPath(*launch, phase.rows, arm, true, b.gate, true); !checked) {
        return Refused(std::format("{} at {} rows: {}", b.gate.name, phase.rows, checked.error()));
      }
      Exl3Op gate_up = op("gate_up", l, Exl3Owner::kExl3, "exl3.multi_linear.mgemm",
                          {"mlp_norm.out"}, {"gate", "up"});
      gate_up.linear = b.gate.name;
      gate_up.second = b.up.name;
      gate_up.launch = *launch;
      ops.push_back(std::move(gate_up));
    } else {
      for (const auto& [proj, lin, out] :
           {std::tuple{"gate_proj", &b.gate, "gate"}, std::tuple{"up_proj", &b.up, "up"}}) {
        auto made = linear(proj, l, *lin, "mlp_norm.out", out, true);
        if (!made) {
          return std::unexpected(made.error());
        }
        ops.push_back(std::move(*made));
      }
    }
    ops.push_back(op("swiglu", l, Exl3Owner::kGgml, "ggml.swiglu", {"gate", "up"}, {"swiglu.f32"}));
    ops.push_back(op("swiglu.cast", l, Exl3Owner::kGgml, convert, {"swiglu.f32"}, {"swiglu.out"}));
    {
      auto made = linear("down_proj", l, b.down, "swiglu.out", "down_proj.out", true);
      if (!made) {
        return std::unexpected(made.error());
      }
      ops.push_back(std::move(*made));
    }
    ops.push_back(op("mlp_residual_add", l, Exl3Owner::kGgml, "ggml.add",
                     {"resid.mid", "down_proj.out"}, {"resid.out"}));
    for (Exl3Op& o : ops) {
      plan.ops.push_back(std::move(o));
    }
  }
  plan.ops.push_back(op("final_norm", -1, Exl3Owner::kGgml, norm, {"resid.out", "final_norm.w"},
                        {"final_norm.f32"}));
  plan.ops.push_back(
      op("final_norm.cast", -1, Exl3Owner::kGgml, convert, {"final_norm.f32"}, {"final_norm.out"}));
  {
    auto made = linear("lm_head", -1, binding.lm_head, "final_norm.out", "logits", false);
    if (!made) {
      return std::unexpected(made.error());
    }
    plan.ops.push_back(std::move(*made));
  }

  // The linears' scratch, as the launchers size it.
  const auto linear_of = [&](const Exl3Op& o) -> const Exl3LinearBinding& {
    if (o.layer < 0) {
      return binding.lm_head;
    }
    const Exl3LayerBinding& b = binding.layers[static_cast<std::size_t>(o.layer)];
    for (const Exl3LinearBinding* l : {&b.q, &b.k, &b.v, &b.o, &b.gate, &b.up, &b.down}) {
      if (l->name == o.linear) {
        return *l;
      }
    }
    return binding.lm_head;
  };
  std::map<std::string, std::uint64_t, std::less<>> scratch;  // "<op>.<part>" -> bytes
  for (const Exl3Op& o : plan.ops) {
    if (o.owner != Exl3Owner::kExl3 || o.linear.empty()) {
      continue;
    }
    const Exl3LinearBinding& l = linear_of(o);
    const auto rows = static_cast<std::uint64_t>(n);
    const auto k = static_cast<std::uint64_t>(l.k);
    switch (o.launch.path) {
      case Exl3Path::kGemm:
      case Exl3Path::kGemv:
        scratch[o.name + ".a_had"] = rows * k * 2;
        break;
      case Exl3Path::kMulti:
        scratch[o.name + ".a_had"] = 2 * rows * k * 2;
        break;
      case Exl3Path::kReconstruct:
        scratch[o.name + ".xh"] = rows * k * 2;
        [[fallthrough]];
      case Exl3Path::kReconstructFused:
        scratch[o.name + ".w"] =
            k * static_cast<std::uint64_t>(std::min(l.n, kExl3SliceColumns)) * 2;
        break;
    }
  }

  // Placement: one timeline of the inputs, the embedding, one layer (its
  // tensors take the same slots in every layer) and the output. The
  // residual stream (embed.out, every layer's resid.in and resid.out)
  // lives in one buffer for the whole phase; the inputs too.
  std::vector<Buffer> buffers;
  std::map<std::string, std::size_t, std::less<>> buffer_of;
  const auto add = [&](std::vector<std::string> names, std::uint64_t bytes) {
    for (const std::string& name : names) {
      buffer_of[name] = buffers.size();
    }
    buffers.push_back(Buffer{.names = std::move(names), .bytes = RoundUp(bytes, kAlignment)});
  };
  for (const auto& [name, t] : plan.tensors) {
    if (name == "embed.out" || name == "resid.in" || name == "resid.out" || name == "gate" ||
        name == "up") {
      continue;
    }
    add({name}, t.bytes());
  }
  add({"embed.out", "resid.in", "resid.out"}, plan.tensors.at("embed.out").bytes());
  const std::uint64_t plane = RoundUp(plan.tensors.at("gate").bytes(), kAlignment);
  add({"gate", "up"}, 2 * plane);
  for (const auto& [name, bytes] : scratch) {
    add({name}, bytes);
  }
  // Steps: the first layer's operations stand for every layer's. A tensor
  // live when the layers begin (the residual stream, the positions and the
  // mask) stays live through the whole layer span, since every layer reads
  // it; everything else lives from its first use to its last.
  int step = 0;
  int layer_first = -1;
  int layer_last = -1;
  const auto touch = [&](const std::string& name) {
    if (const auto found = buffer_of.find(name); found != buffer_of.end()) {
      Buffer& b = buffers[found->second];
      b.first = std::min(b.first, step);
      b.last = std::max(b.last, step);
    }
  };
  for (const Exl3Op& o : plan.ops) {
    if (o.layer > 0) {
      continue;
    }
    if (o.layer == 0) {
      layer_first = layer_first < 0 ? step : layer_first;
      layer_last = step;
    }
    for (const auto& names : {o.inputs, o.outputs}) {
      for (const std::string& name : names) {
        touch(name);
      }
    }
    for (const char* part : {".a_had", ".xh", ".w"}) {
      touch(o.name + part);
    }
    ++step;
  }
  for (Buffer& b : buffers) {
    if (b.first < layer_first && b.last >= layer_first) {
      b.last = std::max(b.last, layer_last);
    }
  }
  // Largest first, each at the lowest offset free for its whole lifetime.
  std::vector<std::size_t> order(buffers.size());
  for (std::size_t i = 0; i < order.size(); ++i) {
    order[i] = i;
  }
  std::ranges::stable_sort(
      order, [&](std::size_t a, std::size_t b) { return buffers[a].bytes > buffers[b].bytes; });
  std::vector<std::pair<std::size_t, std::uint64_t>> placed;  // buffer, offset
  std::vector<std::uint64_t> offset_of(buffers.size());
  for (const std::size_t i : order) {
    const Buffer& b = buffers[i];
    if (b.last < 0) {
      return Refused(std::format("{} is never used", b.names.front()));
    }
    std::vector<std::pair<std::uint64_t, std::uint64_t>> busy;
    for (const auto& [j, at] : placed) {
      if (buffers[j].first <= b.last && b.first <= buffers[j].last) {
        busy.emplace_back(at, at + buffers[j].bytes);
      }
    }
    std::ranges::sort(busy);
    std::uint64_t at = 0;
    for (const auto& [begin, end] : busy) {
      if (at + b.bytes <= begin) {
        break;
      }
      at = std::max(at, end);
    }
    placed.emplace_back(i, at);
    offset_of[i] = at;
    plan.region = std::max(plan.region, at + b.bytes);
  }
  for (std::size_t i = 0; i < buffers.size(); ++i) {
    const Buffer& b = buffers[i];
    if (b.names.size() == 2 && b.names[0] == "gate") {
      plan.slots["gate"] = {.offset = offset_of[i], .bytes = plan.tensors.at("gate").bytes()};
      plan.slots["up"] = {.offset = offset_of[i] + plane, .bytes = plan.tensors.at("up").bytes()};
      continue;
    }
    for (const std::string& name : b.names) {
      const auto t = plan.tensors.find(name);
      plan.slots[name] = {.offset = offset_of[i],
                          .bytes = t != plan.tensors.end() ? t->second.bytes() : b.bytes};
    }
  }

  // The launch data, in order.
  base::Sha256 hash;
  Field(hash, "llmp.exl3.qwen2.launch.v0");
  Field(hash, p.name);
  Field(hash, table.source());
  Field(hash, std::format("arm {} rows {} past {} padded {}", arm == Exl3Arm::kG ? "G" : "O",
                          phase.rows, phase.past, plan.padded));
  for (const Exl3Op& o : plan.ops) {
    std::string line = std::format("{} {} {} [", o.name, o.layer, o.implementation);
    for (const std::string& in : o.inputs) {
      line += in + " ";
    }
    line += "] [";
    for (const std::string& out : o.outputs) {
      line += out + " ";
    }
    line += "]";
    if (!o.linear.empty()) {
      line += std::format(" {} {} {}", o.linear, o.second, Describe(o.launch));
    }
    Field(hash, line);
  }
  plan.launch_digest = hash.Finish();
  return plan;
}

}  // namespace llmp::model
