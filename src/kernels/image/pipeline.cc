// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/image/pipeline.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <format>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "execution/registry.h"
#include "kernels/image/gemm.h"
#include "kernels/image/implementations.h"
#include "kernels/image/ops.h"
#include "model/qwen_image.h"

namespace llmp::kernels::image {
namespace {

namespace md = llmp::model;
using execution::Operation;

std::unexpected<std::string> Error(std::string what) { return std::unexpected(std::move(what)); }

template <typename T>
T* At(std::uint64_t address) {
  return reinterpret_cast<T*>(address);  // NOLINT(performance-no-int-to-ptr)
}

Status Cuda(cudaError_t result, std::string_view what) {
  if (result != cudaSuccess) {
    return Error(std::format("{}: {}", what, cudaGetErrorString(result)));
  }
  return {};
}

std::uint64_t Round(std::uint64_t bytes, std::uint64_t to) { return (bytes + to - 1) / to * to; }

// 256-byte aligned buffers one after another from a base (0 to measure).
class Carve {
 public:
  explicit Carve(std::uint64_t base) : base_(base) {}
  std::uint64_t Take(std::uint64_t bytes) {
    const std::uint64_t at = base_ + used_;
    used_ += Round(std::max<std::uint64_t>(bytes, 256), 256);
    return at;
  }
  std::uint64_t used() const { return used_; }

 private:
  std::uint64_t base_;
  std::uint64_t used_ = 0;
};

std::uint64_t U(std::int64_t v) { return static_cast<std::uint64_t>(v); }

// A role: its name, the implementations it takes (the M3 slice's and the
// speed slice's; the same one twice where it has one), and which each named
// plan chooses.
struct RoleInfo {
  std::string_view name;
  std::array<Impl, 3> takes;
  Impl legacy;
  Impl fast;
};

// One implementation, in either plan.
constexpr RoleInfo Only(std::string_view name, Impl impl) {
  return {.name = name, .takes = {impl, impl, impl}, .legacy = impl, .fast = impl};
}
// The M3 slice's implementation, then the speed slice's.
constexpr RoleInfo Two(std::string_view name, Impl legacy, Impl fast) {
  return {.name = name, .takes = {legacy, fast, fast}, .legacy = legacy, .fast = fast};
}
// The M3 slice's, another the role takes, then the speed slice's.
constexpr RoleInfo Three(std::string_view name, Impl legacy, Impl other, Impl fast) {
  return {.name = name, .takes = {legacy, other, fast}, .legacy = legacy, .fast = fast};
}

constexpr std::array<RoleInfo, kRoles> kRoleTable = {{
    Only("text.embed", Impl::kEmbedRows),
    Only("text.norm", Impl::kRmsNorm),
    Two("text.linear", Impl::kLinearCublas, Impl::kLinearCublasLt),
    Only("text.head_norm_rope", Impl::kHeadNormRopeNeox),
    Only("text.attention", Impl::kAttentionShort),
    Only("text.add", Impl::kAdd),
    Only("text.swiglu", Impl::kSwiGlu),
    Two("dit.linear", Impl::kLinearCublas, Impl::kLinearCublasLt),
    Only("dit.silu", Impl::kSilu),
    Only("dit.gelu", Impl::kGeluTanh),
    Only("dit.text_norm", Impl::kRmsNormZeroCentre),
    Only("dit.norm", Impl::kLayerNormModulate),
    Only("dit.head_norm_rope", Impl::kHeadNormRopeComplex),
    Only("dit.text_attention", Impl::kAttentionShort),
    Three("dit.attention", Impl::kFlashAttention, Impl::kFlashAttentionPrefixed,
          Impl::kFlashAttentionNormQ),
    Two("dit.residual", Impl::kGatedResidual, Impl::kGatedResidualNorm),
    Only("dit.swiglu", Impl::kSwiGlu),
    Only("dit.euler", Impl::kEulerStep),
    Only("vae.convert", Impl::kConvert),
    Two("vae.convert_3x3", Impl::kConvert, Impl::kConvertKrsc),
    Only("vae.transpose", Impl::kTranspose),
    Only("vae.scale_shift", Impl::kScaleShift),
    Two("vae.conv_3x3", Impl::kConvIm2col, Impl::kConvImplicit),
    Only("vae.conv_1x1", Impl::kConvProduct),
    Only("vae.norm", Impl::kChannelNorm),
    Only("vae.add", Impl::kAdd),
    Only("vae.upsample", Impl::kUpsample),
    Only("vae.dup_up", Impl::kDupUpAdd),
    Only("vae.attention", Impl::kMatmulSingleHead),
    Only("vae.soft_max", Impl::kSoftMax),
}};

const std::array<RoleInfo, kRoles>& Roles() { return kRoleTable; }

// The operation each implementation declares (implementations.cc's table).
std::expected<Operation, std::string> OperationOf(std::string_view name) {
  static const std::vector<execution::Implementation> declared = Implementations();
  for (const auto& d : declared) {
    if (d.name == name) {
      return d.operation;
    }
  }
  return Error(std::format("no image implementation {}", name));
}

}  // namespace

std::string_view RoleName(Role role) {
  const auto i = static_cast<std::size_t>(role);
  return i < kRoles ? Roles().at(i).name : std::string_view("unknown");
}

std::expected<std::vector<execution::Choice>, std::string> QwenImageChoices(
    PlanKind kind, std::span<const std::string> overrides) {
  std::vector<std::string> names;
  for (const RoleInfo& r : Roles()) {
    names.emplace_back(ImplName(kind == PlanKind::kFast ? r.fast : r.legacy));
  }
  for (const std::string& o : overrides) {
    const auto eq = o.find('=');
    bool found = false;
    for (std::size_t i = 0; eq != std::string::npos && i < kRoles; ++i) {
      if (Roles().at(i).name == std::string_view(o).substr(0, eq)) {
        names.at(i) = o.substr(eq + 1);
        found = true;
      }
    }
    if (!found) {
      return Error(std::format("{}: not role=implementation", o));
    }
  }
  std::vector<execution::Choice> choices;
  for (const std::string& name : names) {
    auto op = OperationOf(name);
    if (!op) {
      return std::unexpected(op.error());
    }
    choices.push_back({.operation = *op, .implementation = name});
  }
  return choices;
}

// ---------------------------------------------------------------- layouts

TextMemory TextLayout(std::uint64_t base, const md::QwenImageTextProfile& p, std::int64_t rows,
                      std::uint64_t& bytes) {
  Carve c(base);
  const std::int64_t width = p.width;
  const std::int64_t q = std::int64_t{p.heads} * p.head_dim;
  const std::int64_t kv = std::int64_t{p.kv_heads} * p.head_dim;
  TextMemory m;
  m.x = c.Take(U(rows * width * 2));
  m.n = c.Take(U(rows * width * 2));
  m.q = c.Take(U(rows * q * 2));
  m.k = c.Take(U(rows * kv * 2));
  m.v = c.Take(U(rows * kv * 2));
  m.attn = c.Take(U(rows * q * 2));
  m.o = c.Take(U(rows * width * 2));
  m.g = c.Take(U(rows * std::int64_t{p.ffn} * 2));
  m.u = c.Take(U(rows * std::int64_t{p.ffn} * 2));
  m.ids = c.Take(U(rows * 4));
  m.bad = c.Take(4);
  m.cos = c.Take(U(rows * 256));
  m.sin = c.Take(U(rows * 256));
  m.rows = rows;
  bytes = c.used();
  return m;
}

DitMemory DitLayout(std::uint64_t base, const md::QwenImageDenoiserProfile& p, std::int64_t rows,
                    std::uint64_t& bytes) {
  Carve c(base);
  const std::int64_t width = p.width;
  DitMemory m;
  m.x = c.Take(U(rows * width * 2));
  m.n = c.Take(U(rows * width * 2));
  m.q = c.Take(U(rows * width * 2));
  m.k = c.Take(U(rows * width * 2));
  m.v = c.Take(U(rows * width * 2));
  m.attn = c.Take(U(rows * width * 2));
  m.o = c.Take(U(rows * width * 2));
  m.g = c.Take(U(rows * std::int64_t{p.mlp} * 2));
  m.u = c.Take(U(rows * std::int64_t{p.mlp} * 2));
  m.sin = c.Take(U(2 * std::int64_t{p.timestep_dim} * 2));
  m.dt = c.Take(4);
  m.t1 = c.Take(U(2 * width * 2));
  m.t2 = c.Take(U(2 * width * 2));
  m.temb = c.Take(U(2 * width * 2));
  m.mod = c.Take(U(width * 2 * 4 * 2));
  m.nscale = c.Take(U(2 * width * 2));
  bytes = c.used();
  return m;
}

DitMemory OwnLayout(std::uint64_t base, const md::QwenImageDenoiserProfile& p, std::int64_t text,
                    std::int64_t image, std::uint64_t& bytes) {
  Carve c(base);
  const std::int64_t width = p.width;
  DitMemory m;
  m.embeds = c.Take(U(text * p.context * 2));
  m.txt = c.Take(U(text * width * 2));
  m.pk = c.Take(U(std::int64_t{p.blocks} * text * width * 2));
  m.pv = c.Take(U(std::int64_t{p.blocks} * text * width * 2));
  m.freqs = c.Take(U((text + image) * 128 * 4));
  m.latents = c.Take(U(image * p.in_channels * 2));
  m.noise = c.Take(U(image * p.out_channels * 2));
  m.text = text;
  m.image = image;
  bytes = c.used();
  return m;
}

VaeMemory VaeLayout(std::uint64_t base, const md::QwenImageVaeProfile& p, std::uint32_t grid,
                    std::span<const std::uint64_t> f32_bytes, bool bf16_weights,
                    std::vector<std::uint64_t>& weights, std::uint64_t& bytes) {
  Carve c(base);
  VaeMemory m;
  weights.clear();
  if (bf16_weights) {
    for (const std::uint64_t b : f32_bytes) {
      weights.push_back(c.Take(b / 2));
    }
  }
  const auto steps = md::VaeDecoderPlan(p, grid, grid);
  std::array<std::uint64_t, md::kVaeBuffers> need{};
  std::uint64_t attention_pixels = 0;
  for (const md::VaeStep& st : steps) {
    const std::uint64_t in = std::uint64_t{st.in_channels} * st.height * st.width;
    const std::uint64_t scale = st.kind == md::VaeStep::Kind::kUpsample ? 4 : 1;
    const std::uint64_t out = std::uint64_t{st.out_channels} * st.height * st.width * scale;
    need.at(st.in) =
        std::max(need.at(st.in), st.kind == md::VaeStep::Kind::kAttention ? 3 * in : in);
    need.at(st.out) =
        std::max(need.at(st.out), st.kind == md::VaeStep::Kind::kAddDupUp ? 4 * out : out);
    need.at(st.aux) = std::max(need.at(st.aux), in);
    if (st.kind == md::VaeStep::Kind::kAttention) {
      attention_pixels = std::max(attention_pixels, std::uint64_t{st.height} * st.width);
    }
  }
  need.at(md::kVaeX) =
      std::max<std::uint64_t>(need.at(md::kVaeX), std::uint64_t{p.z_dim} * grid * grid);
  for (std::size_t i = 0; i < need.size(); ++i) {
    m.buf.at(i) = c.Take(need.at(i) * 2);
  }
  m.stats = c.Take(std::uint64_t{2} * 64 * 2);
  m.col = c.Take(kIm2colBytes);
  m.col_bytes = kIm2colBytes;
  m.scores_bytes = attention_pixels * attention_pixels * 4;
  m.scores = c.Take(m.scores_bytes);
  m.probs = c.Take(attention_pixels * attention_pixels * 2);
  m.grid = grid;
  bytes = c.used();
  return m;
}

// ---------------------------------------------------------------- binding

QwenImagePipeline::~QwenImagePipeline() = default;

std::expected<std::unique_ptr<QwenImagePipeline>, std::string> QwenImagePipeline::Bind(
    const execution::Registry& registry, const execution::Plan& plan,
    const md::QwenImageProfile& profile) {
  auto resolved = execution::Resolve(registry, plan);
  if (!resolved) {
    return Error(std::format("the plan does not resolve at operation {}: {}",
                             resolved.error().operation, resolved.error().detail));
  }
  if (resolved->size() != kRoles) {
    return Error(std::format("the plan has {} operations; the pipeline has {} roles",
                             resolved->size(), kRoles));
  }
  auto p = std::unique_ptr<QwenImagePipeline>(new QwenImagePipeline());  // NOLINT
  p->profile_ = profile;
  for (std::size_t i = 0; i < kRoles; ++i) {
    auto impl = image::Bind(resolved->at(i));
    if (!impl) {
      return Error(std::format("{}: {}", Roles().at(i).name, impl.error()));
    }
    const auto& takes = Roles().at(i).takes;
    if (std::ranges::find(takes, *impl) == takes.end()) {
      return Error(std::format("{} does not take {}", Roles().at(i).name, ImplName(*impl)));
    }
    p->bound_.at(i) = *impl;
  }
  const bool implicit = p->bound(Role::kVaeConv3x3) == Impl::kConvImplicit;
  const bool krsc = p->bound(Role::kVaeConvert3x3) == Impl::kConvertKrsc;
  if (implicit != krsc) {
    return Error("vae.conv_3x3 and vae.convert_3x3 disagree on the weights' layout");
  }
  // The plan's identity, and which VAE tensors are 3x3 weights.
  p->identity_ = resolved->identity();
  const auto vae_tensors = md::VaeDecoderTensors(profile.vae);
  p->conv3x3_.assign(vae_tensors.size(), {0, 0});
  for (const md::QwenImageTensor& t : vae_tensors) {
    std::uint64_t n = 1;
    for (const std::uint64_t d : t.shape) {
      n *= d;
    }
    p->vae_elements_.push_back(n);
  }
  for (const md::VaeStep& st : md::VaeDecoderPlan(profile.vae, 1, 1)) {
    if (st.kind == md::VaeStep::Kind::kConv3x3 && st.weight >= 0) {
      p->conv3x3_.at(static_cast<std::size_t>(st.weight)) = {st.out_channels, st.in_channels};
    }
  }
  return p;
}

std::string QwenImagePipeline::Describe() const {
  std::string out = "{";
  for (std::size_t i = 0; i < kRoles; ++i) {
    out += std::format(R"({}"{}": "{}")", i == 0 ? "" : ", ", Roles().at(i).name,
                       ImplName(bound_.at(i)));
  }
  return out + "}";
}

Status QwenImagePipeline::VaeWeightBf16(std::size_t tensor, std::span<const std::byte> f32,
                                        std::span<Bf16> out) const {
  const std::size_t n = f32.size() / 4;
  if (f32.size() % 4 != 0 || out.size() != n) {
    return Error("VaeWeightBf16: sizes");
  }
  const auto value = [&](std::size_t i) {
    float v = 0;
    std::memcpy(&v, f32.data() + (i * 4), 4);
    return md::ToBf16(v);
  };
  const auto [co, ci] =
      tensor < conv3x3_.size() ? conv3x3_.at(tensor) : std::array<std::uint32_t, 2>{0, 0};
  if (co != 0 && bound(Role::kVaeConvert3x3) == Impl::kConvertKrsc) {
    if (n != std::size_t{co} * ci * 9) {
      return Error("VaeWeightBf16: a 3x3 convolution's weights");
    }
    // [co][ci][tap] -> [co][tap][ci]
    std::size_t i = 0;
    for (std::size_t o = 0; o < co; ++o) {
      for (std::size_t c = 0; c < ci; ++c) {
        for (std::size_t tap = 0; tap < 9; ++tap) {
          out[(((o * 9) + tap) * ci) + c] = value(i++);
        }
      }
    }
    return {};
  }
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = value(i);
  }
  return {};
}

Status QwenImagePipeline::Linear(Role role, const Handles& h, std::uint64_t x, std::int64_t ldx,
                                 std::uint64_t w, std::int64_t ldw, std::uint64_t out,
                                 std::int64_t ldo, std::int64_t m, std::int64_t n, std::int64_t k,
                                 Stream s) const {
  if (bound(role) == Impl::kLinearCublasLt) {
    if (h.lt == nullptr) {
      return Error("image.linear.cublaslt without a cuBLASLt handle");
    }
    return h.lt->Linear(At<Bf16>(x), ldx, At<Bf16>(w), ldw, At<Bf16>(out), ldo, m, n, k, s);
  }
  if (h.blas == nullptr) {
    return Error("image.linear.cublas without a cuBLAS handle");
  }
  return image::Linear(h.blas, At<Bf16>(x), ldx, At<Bf16>(w), ldw, At<Bf16>(out), ldo, m, n, k);
}

// ---------------------------------------------------------------- encode

Status QwenImagePipeline::Encode(const TextMemory& m, const Handles& h, Stream s) const {
  const md::QwenImageTextProfile& p = profile_.text;
  const std::int64_t rows = m.rows;
  const std::int64_t width = p.width;
  const std::int64_t q = std::int64_t{p.heads} * p.head_dim;
  const std::int64_t kv = std::int64_t{p.kv_heads} * p.head_dim;
  const auto per_layer = static_cast<std::size_t>(md::TextTensor::kCount);
  if (m.tensors.size() != 1 + (per_layer * p.layers) || rows <= m.drop || m.drop < 0) {
    return Error("Encode: the text encoder's tensors or rows");
  }
  const auto t = [&](std::size_t layer, md::TextTensor which) {
    return m.tensors[1 + (layer * per_layer) + static_cast<std::size_t>(which)];
  };
  Status r = EmbedRows(At<Bf16>(m.tensors[0]), p.vocab, At<std::int32_t>(m.ids), rows, width,
                       At<Bf16>(m.x), At<std::int32_t>(m.bad), s);
  const float scale = 1.0f / std::sqrt(static_cast<float>(p.head_dim));
  for (std::size_t l = 0; r && l < p.layers; ++l) {
    using T = md::TextTensor;
    r = RmsNorm(At<Bf16>(m.x), At<Bf16>(t(l, T::kInputNorm)), At<Bf16>(m.n), rows, width, p.rms_eps,
                s);
    r = r ? Linear(Role::kTextLinear, h, m.n, width, t(l, T::kQ), width, m.q, q, rows, q, width, s)
          : r;
    r = r ? Linear(Role::kTextLinear, h, m.n, width, t(l, T::kK), width, m.k, kv, rows, kv, width,
                   s)
          : r;
    r = r ? Linear(Role::kTextLinear, h, m.n, width, t(l, T::kV), width, m.v, kv, rows, kv, width,
                   s)
          : r;
    r = r ? HeadNormRopeNeox(At<Bf16>(m.q), q, rows, p.heads, At<Bf16>(t(l, T::kQNorm)),
                             At<Bf16>(m.cos), At<Bf16>(m.sin), p.rms_eps, s)
          : r;
    r = r ? HeadNormRopeNeox(At<Bf16>(m.k), kv, rows, p.kv_heads, At<Bf16>(t(l, T::kKNorm)),
                             At<Bf16>(m.cos), At<Bf16>(m.sin), p.rms_eps, s)
          : r;
    r = r ? SmallAttention(At<Bf16>(m.q), q, At<Bf16>(m.k), kv, At<Bf16>(m.v), kv, At<Bf16>(m.attn),
                           q, rows, p.heads, p.kv_heads, true, scale, s)
          : r;
    r = r ? Linear(Role::kTextLinear, h, m.attn, q, t(l, T::kO), q, m.o, width, rows, width, q, s)
          : r;
    r = r ? Add(At<Bf16>(m.x), At<Bf16>(m.o), At<Bf16>(m.x), rows * width, s) : r;
    r = r ? RmsNorm(At<Bf16>(m.x), At<Bf16>(t(l, T::kPostNorm)), At<Bf16>(m.n), rows, width,
                    p.rms_eps, s)
          : r;
    r = r ? Linear(Role::kTextLinear, h, m.n, width, t(l, T::kGate), width, m.g, p.ffn, rows, p.ffn,
                   width, s)
          : r;
    r = r ? Linear(Role::kTextLinear, h, m.n, width, t(l, T::kUp), width, m.u, p.ffn, rows, p.ffn,
                   width, s)
          : r;
    r = r ? SwiGlu(At<Bf16>(m.g), p.ffn, At<Bf16>(m.u), p.ffn, At<Bf16>(m.g), rows, p.ffn, s) : r;
    r = r ? Linear(Role::kTextLinear, h, m.g, p.ffn, t(l, T::kDown), p.ffn, m.o, width, rows, width,
                   p.ffn, s)
          : r;
    r = r ? Add(At<Bf16>(m.x), At<Bf16>(m.o), At<Bf16>(m.x), rows * width, s) : r;
  }
  const std::int64_t kept = rows - m.drop;
  return r ? Cuda(cudaMemcpyAsync(At<void>(m.embeds), At<Bf16>(m.x) + (m.drop * width),
                                  static_cast<std::size_t>(kept * width * 2),
                                  cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(s)),
                  "the kept rows")
           : r;
}

// ---------------------------------------------------------------- denoise

namespace {

Bf16* Row(std::uint64_t buf, std::int64_t row, std::int64_t width) {
  return At<Bf16>(buf) + (row * width);
}

}  // namespace

Status QwenImagePipeline::TextRows(const DitMemory& m, const Handles& h, Stream s) const {
  const md::QwenImageDenoiserProfile& p = profile_.denoiser;
  const std::int64_t width = p.width;
  const auto g = [&](md::DenoiserGlobal which) {
    return m.tensors[static_cast<std::size_t>(which)];
  };
  Status r = ZeroCenterRmsNorm(At<Bf16>(m.embeds), At<Bf16>(g(md::DenoiserGlobal::kTxtNorm)),
                               At<Bf16>(m.n), m.text, p.context, p.eps, s);
  r = r ? Linear(Role::kDitLinear, h, m.n, p.context, g(md::DenoiserGlobal::kTxtIn), p.context, m.q,
                 width, m.text, width, p.context, s)
        : r;
  r = r ? GeluTanh(At<Bf16>(m.q), At<Bf16>(m.q), m.text * width, s) : r;
  r = r ? Linear(Role::kDitLinear, h, m.q, width, g(md::DenoiserGlobal::kTxtOut), width, m.txt,
                 width, m.text, width, width, s)
        : r;
  return r;
}

Status QwenImagePipeline::Step(bool first, const DitMemory& m, const Handles& h, Stream s) const {
  const md::QwenImageDenoiserProfile& p = profile_.denoiser;
  const std::size_t tensors =
      static_cast<std::size_t>(md::DenoiserGlobal::kCount) +
      (std::size_t{p.blocks} * static_cast<std::size_t>(md::BlockTensor::kCount));
  if (m.tensors.size() != tensors || m.text <= 0 || m.image <= 0) {
    return Error("Step: the denoiser's tensors or rows");
  }
  const std::int64_t width = p.width;
  const auto g = [&](md::DenoiserGlobal which) {
    return m.tensors[static_cast<std::size_t>(which)];
  };
  using G = md::DenoiserGlobal;
  auto* const cs = static_cast<cudaStream_t>(s);
  Status r = Linear(Role::kDitLinear, h, m.sin, p.timestep_dim, g(G::kTime1), p.timestep_dim, m.t1,
                    width, 2, width, p.timestep_dim, s);
  r = r ? Silu(At<Bf16>(m.t1), At<Bf16>(m.t1), 2 * width, s) : r;
  r = r ? Linear(Role::kDitLinear, h, m.t1, width, g(G::kTime2), width, m.temb, width, 2, width,
                 width, s)
        : r;
  r = r ? Silu(At<Bf16>(m.temb), At<Bf16>(m.t2), 2 * width, s) : r;
  r = r ? Linear(Role::kDitLinear, h, m.t2, width, g(G::kModulation), width, m.mod, 4 * width, 2,
                 4 * width, width, s)
        : r;
  r = r ? Linear(Role::kDitLinear, h, m.t2, width, g(G::kNormOut), width, m.nscale, width, 2, width,
                 width, s)
        : r;
  if (r && first) {
    r = Cuda(
        cudaMemcpyAsync(At<void>(m.x), At<void>(m.txt),
                        static_cast<std::size_t>(m.text * width * 2), cudaMemcpyDeviceToDevice, cs),
        "the text rows");
  }
  r = r ? Linear(Role::kDitLinear, h, m.latents, p.in_channels, g(G::kImgIn), p.in_channels,
                 m.x + static_cast<std::uint64_t>(m.text * width * 2), width, m.image, width,
                 p.in_channels, s)
        : r;
  for (std::uint32_t b = 0; r && b < p.blocks; ++b) {
    r = Block(b, first, m, h, s);
  }
  // norm_out on the image rows (fused into the last block's residual
  // otherwise), then proj_out.
  if (r && bound(Role::kDitResidual) != Impl::kGatedResidualNorm) {
    r = LayerNormModulate(Row(m.x, m.text, width), Row(m.n, m.text, width), m.image, width, p.eps,
                          At<Bf16>(m.nscale), width, 0, s);
  }
  r = r ? Linear(Role::kDitLinear, h, m.n + static_cast<std::uint64_t>(m.text * width * 2), width,
                 g(G::kProjOut), width, m.noise, p.out_channels, m.image, p.out_channels, width, s)
        : r;
  // FlowMatchEulerDiscreteScheduler.step, dt (BF16) from the device.
  r = r ? EulerStepAt(At<Bf16>(m.latents), At<Bf16>(m.noise), At<Bf16>(m.latents), At<float>(m.dt),
                      m.image * p.in_channels, s)
        : r;
  return r;
}

Status QwenImagePipeline::Block(std::uint32_t b, bool first, const DitMemory& m, const Handles& h,
                                Stream s) const {
  const md::QwenImageDenoiserProfile& p = profile_.denoiser;
  using T = md::BlockTensor;
  const std::int64_t width = p.width;
  const std::int64_t mlp = p.mlp;
  const std::int64_t mstride = 4 * width;
  const std::int64_t rows = first ? m.text + m.image : m.image;
  const std::int64_t row0 = first ? 0 : m.text;
  const std::int64_t first_target = first ? m.text : 0;
  const bool fused = bound(Role::kDitResidual) == Impl::kGatedResidualNorm;
  auto* const cs = static_cast<cudaStream_t>(s);
  const auto w = [&](T which) {
    return m.tensors[static_cast<std::size_t>(md::DenoiserGlobal::kCount) +
                     (b * static_cast<std::size_t>(T::kCount)) + static_cast<std::size_t>(which)];
  };
  const Bf16* mod = At<Bf16>(m.mod);
  Bf16* x = Row(m.x, row0, width);
  const auto at = [&](std::uint64_t buf, std::int64_t row, std::int64_t cols) {
    return buf + static_cast<std::uint64_t>(row * cols * 2);
  };
  // The block's first norm: from the previous block's fused residual, but
  // for the first block.
  Status r;
  if (!fused || b == 0) {
    r = LayerNormModulate(x, Row(m.n, row0, width), rows, width, p.eps, mod, mstride, first_target,
                          s);
  }
  r = r ? Linear(Role::kDitLinear, h, at(m.n, row0, width), width, w(T::kQ), width,
                 at(m.q, row0, width), width, rows, width, width, s)
        : r;
  r = r ? Linear(Role::kDitLinear, h, at(m.n, row0, width), width, w(T::kK), width,
                 at(m.k, row0, width), width, rows, width, width, s)
        : r;
  r = r ? Linear(Role::kDitLinear, h, at(m.n, row0, width), width, w(T::kV), width,
                 at(m.v, row0, width), width, rows, width, width, s)
        : r;
  // Attention: q and k normed and rotated in place (q in the attention's
  // registers past the first step, with norm_q), the prefix cache in BF16.
  const Impl attention = bound(Role::kDitAttention);
  const bool norm_q = !first && attention == Impl::kFlashAttentionNormQ;
  if (!norm_q) {
    r = r ? HeadNormRopeComplex(Row(m.q, row0, width), width, rows, p.heads, At<Bf16>(w(T::kQNorm)),
                                At<float>(m.freqs) + (row0 * 128), p.eps, s)
          : r;
  }
  r = r ? HeadNormRopeComplex(Row(m.k, row0, width), width, rows, p.heads, At<Bf16>(w(T::kKNorm)),
                              At<float>(m.freqs) + (row0 * 128), p.eps, s)
        : r;
  const auto prefix = static_cast<std::size_t>(m.text * width * 2);
  Bf16* pk = At<Bf16>(m.pk) + (static_cast<std::int64_t>(b) * m.text * width);
  Bf16* pv = At<Bf16>(m.pv) + (static_cast<std::int64_t>(b) * m.text * width);
  const float scale = 1.0f / std::sqrt(128.0f);
  if (r && first) {
    r = Cuda(cudaMemcpyAsync(pk, At<void>(m.k), prefix, cudaMemcpyDeviceToDevice, cs), "cache");
    r = r ? Cuda(cudaMemcpyAsync(pv, At<void>(m.v), prefix, cudaMemcpyDeviceToDevice, cs), "cache")
          : r;
    r = r ? SmallAttention(At<Bf16>(m.q), width, At<Bf16>(m.k), width, At<Bf16>(m.v), width,
                           At<Bf16>(m.attn), width, m.text, p.heads, p.heads, true, scale, s)
          : r;
  } else if (r && attention == Impl::kFlashAttention) {
    r = Cuda(cudaMemcpyAsync(At<void>(m.k), pk, prefix, cudaMemcpyDeviceToDevice, cs), "cache");
    r = r ? Cuda(cudaMemcpyAsync(At<void>(m.v), pv, prefix, cudaMemcpyDeviceToDevice, cs), "cache")
          : r;
  }
  if (!first && attention != Impl::kFlashAttention) {
    // The image rows' K and V follow the (uncopied) text rows.
    const QueryNorm q_norm{.weight = At<Bf16>(w(T::kQNorm)),
                           .freqs = At<float>(m.freqs) + (m.text * 128),
                           .eps = p.eps};
    r = r ? FlashAttentionPrefixed(Row(m.q, m.text, width), width, pk, pv, m.text,
                                   Row(m.k, m.text, width), width, Row(m.v, m.text, width), width,
                                   Row(m.attn, m.text, width), width, m.image, m.text + m.image,
                                   p.heads, p.heads, scale, s, norm_q ? &q_norm : nullptr)
          : r;
  } else {
    r = r ? FlashAttention(Row(m.q, m.text, width), width, At<Bf16>(m.k), width, At<Bf16>(m.v),
                           width, Row(m.attn, m.text, width), width, m.image, m.text + m.image,
                           p.heads, p.heads, scale, s)
          : r;
  }
  // Out projection, gated residual (and the MLP's norm), the MLP.
  r = r ? Linear(Role::kDitLinear, h, at(m.attn, row0, width), width, w(T::kOut), width,
                 at(m.o, row0, width), width, rows, width, width, s)
        : r;
  if (fused) {
    // y's rows are `width` apart, the modulation's `mstride`.
    // NOLINTNEXTLINE(readability-suspicious-call-argument)
    r = r ? GatedResidualNorm(x, Row(m.o, row0, width), width, rows, width, mod + width, mstride,
                              first_target, Row(m.n, row0, width), p.eps, mod + (2 * width),
                              mstride, s)
          : r;
  } else {
    r = r ? GatedResidual(x, Row(m.o, row0, width), width, rows, width, mod + width, mstride,
                          first_target, s)
          : r;
    r = r ? LayerNormModulate(x, Row(m.n, row0, width), rows, width, p.eps, mod + (2 * width),
                              mstride, first_target, s)
          : r;
  }
  r = r ? Linear(Role::kDitLinear, h, at(m.n, row0, width), width, w(T::kGate), width,
                 at(m.g, row0, mlp), mlp, rows, mlp, width, s)
        : r;
  r = r ? Linear(Role::kDitLinear, h, at(m.n, row0, width), width, w(T::kProj), width,
                 at(m.u, row0, mlp), mlp, rows, mlp, width, s)
        : r;
  r = r ? SwiGlu(Row(m.g, row0, mlp), mlp, Row(m.u, row0, mlp), mlp, Row(m.g, row0, mlp), rows, mlp,
                 s)
        : r;
  r = r ? Linear(Role::kDitLinear, h, at(m.g, row0, mlp), mlp, w(T::kMlpOut), mlp,
                 at(m.o, row0, width), width, rows, width, mlp, s)
        : r;
  if (fused) {
    // The next block's first norm, or norm_out (its scale nscale, one row
    // per modulation row; the text rows at the first step are normed too
    // and never read).
    const bool last = b + 1 == p.blocks;
    // NOLINTNEXTLINE(readability-suspicious-call-argument)
    r = r ? GatedResidualNorm(x, Row(m.o, row0, width), width, rows, width, mod + (3 * width),
                              mstride, first_target, Row(m.n, row0, width), p.eps,
                              last ? At<Bf16>(m.nscale) : mod, last ? width : mstride, s)
          : r;
  } else {
    r = r ? GatedResidual(x, Row(m.o, row0, width), width, rows, width, mod + (3 * width), mstride,
                          first_target, s)
          : r;
  }
  return r;
}

// ---------------------------------------------------------------- decode

Status QwenImagePipeline::ConvertVae(const VaeMemory& m, Stream s) const {
  if (m.f32.size() != m.weights.size() || m.f32.size() != conv3x3_.size() ||
      m.f32.size() != vae_elements_.size()) {
    return Error("ConvertVae: the VAE's tensors");
  }
  Status r;
  for (std::size_t i = 0; r && i < m.f32.size(); ++i) {
    const auto [co, ci] = conv3x3_.at(i);
    if (co != 0 && bound(Role::kVaeConvert3x3) == Impl::kConvertKrsc) {
      r = Conv3x3WeightsKrsc(At<float>(m.f32[i]), At<Bf16>(m.weights[i]), co, ci, s);
    } else {
      r = F32ToBf16(At<float>(m.f32[i]), At<Bf16>(m.weights[i]),
                    static_cast<std::int64_t>(vae_elements_.at(i)), s);
    }
  }
  return r;
}

Status QwenImagePipeline::Decode(const VaeMemory& m, const Handles& h, Stream s) const {
  const md::QwenImageVaeProfile& p = profile_.vae;
  const std::uint32_t grid = m.grid;
  const auto steps = md::VaeDecoderPlan(p, grid, grid);
  auto* const cs = static_cast<cudaStream_t>(s);
  if (m.weights.size() != conv3x3_.size()) {
    return Error("Decode: the VAE's weights");
  }
  const auto buf = [&](std::uint8_t i) { return At<Bf16>(m.buf.at(i)); };
  const auto weight = [&](std::int32_t i) {
    return At<Bf16>(m.weights[static_cast<std::size_t>(i)]);
  };
  const std::int64_t pixels0 = std::int64_t{grid} * grid;
  // _unpack_latents, then latents * std + mean in BF16.
  Status r = Transpose(At<Bf16>(m.latents), buf(md::kVaeH), pixels0, p.z_dim, s);
  r = r ? ScaleShiftChannels(buf(md::kVaeH), At<Bf16>(m.stats), At<Bf16>(m.stats) + 64,
                             buf(md::kVaeX), p.z_dim, pixels0, s)
        : r;
  for (const md::VaeStep& step : steps) {
    if (!r) {
      break;
    }
    using K = md::VaeStep::Kind;
    const std::int64_t pixels = std::int64_t{step.height} * step.width;
    Bf16* in = buf(step.in);
    Bf16* out = buf(step.out);
    Bf16* aux = buf(step.aux);
    switch (step.kind) {
      case K::kConv1x1:
        r = FillBias(weight(step.bias), out, step.out_channels, pixels, s);
        r = r ? ConvProduct(h.blas, weight(step.weight), step.out_channels, step.in_channels, in,
                            pixels, out, pixels, pixels, true)
              : r;
        break;
      case K::kConv3x3: {
        if (bound(Role::kVaeConv3x3) == Impl::kConvImplicit) {
          r = Conv3x3Implicit(in, step.in_channels, step.height, step.width, weight(step.weight),
                              weight(step.bias), out, step.out_channels, s);
          break;
        }
        r = FillBias(weight(step.bias), out, step.out_channels, pixels, s);
        const std::int64_t inner = std::int64_t{step.in_channels} * 9;
        std::int64_t chunk = static_cast<std::int64_t>(m.col_bytes / 2) / inner;
        chunk = std::min<std::int64_t>(pixels, chunk / step.width * step.width);
        if (chunk <= 0) {
          r = Error("im2col buffer smaller than one image row");
        }
        for (std::int64_t p0 = 0; r && p0 < pixels; p0 += chunk) {
          const std::int64_t n = std::min(chunk, pixels - p0);
          r = Im2Col3x3(in, step.in_channels, step.height, step.width, p0, n, At<Bf16>(m.col), s);
          // col's rows are `n` long, out's `pixels`, and `n` pixels.
          // NOLINTNEXTLINE(readability-suspicious-call-argument)
          r = r ? ConvProduct(h.blas, weight(step.weight), step.out_channels, inner,
                              At<Bf16>(m.col), n, out + p0, pixels, n, true)
                : r;
        }
        break;
      }
      case K::kNormSilu:
      case K::kNorm:
        r = ChannelRmsNorm(in, weight(step.weight), out, step.in_channels, pixels,
                           step.kind == K::kNormSilu, s);
        break;
      case K::kAdd:
        r = Add(in, aux, out, std::int64_t{step.out_channels} * pixels, s);
        break;
      case K::kCopy:
        r = Cuda(cudaMemcpyAsync(out, in, static_cast<std::size_t>(step.in_channels * pixels * 2),
                                 cudaMemcpyDeviceToDevice, cs),
                 "a block input");
        break;
      case K::kUpsample:
        r = Upsample2x(in, out, step.in_channels, step.height, step.width, s);
        break;
      case K::kAttention: {
        const std::int64_t ch = step.in_channels;
        if (std::cmp_greater(pixels * pixels * 4, m.scores_bytes)) {
          r = Error("the VAE attention's scores exceed their buffer");
          break;
        }
        r = ScoresQtK(h.blas, in, in + (ch * pixels), At<float>(m.scores), ch, pixels, pixels);
        r = r ? SoftmaxRowsToBf16(At<float>(m.scores), At<Bf16>(m.probs), pixels, pixels,
                                  1.0f / std::sqrt(static_cast<float>(ch)), s)
              : r;
        r = r ? ValuesTimesProbs(h.blas, in + (2 * ch * pixels), At<Bf16>(m.probs), out, ch, pixels,
                                 pixels)
              : r;
        break;
      }
      case K::kAddDupUp:
        r = AddDupUp(out, aux, step.in_channels, step.out_channels, step.factor_t, step.height,
                     step.width, s);
        break;
    }
  }
  return r;
}

}  // namespace llmp::kernels::image
