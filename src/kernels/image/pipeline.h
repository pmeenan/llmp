// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The Qwen-Image-2.1 pipeline's three phases under llmpalooza's dispatch (D-053;
// docs/experiments/qwen-image-native/README.md), CUDA builds only: the text
// encoder (Qwen3-VL's text path), a denoising step of the block-causal DiT
// with its prefix K/V cache and the flow-matching Euler update, and the VAE
// decoder, each queued on one stream over device memory the caller lays out
// (the resident harness in cudaMalloc memory, the paged runner in device
// VMM; benchmarks/qwen_image_exec.cc, qwen_image_runner.h).
//
// A plan names one implementation for each of the pipeline's roles (Role,
// in order), from this module's declarations (implementations.h). Bind
// resolves it against the registry and binds each role to the
// implementation it names, identity and all, or refuses the plan: an
// implementation the build lacks or holds with another identity, or one a
// role does not take; no other implementation is ever bound in its place.
// Every launch then dispatches on the role's bound implementation.
//
// Two plans are named here: the M3 slice's (kLegacy: cublasGemmEx, the
// gated residual and the norm apart, im2col convolutions) and the speed
// slice's (kFast: pinned cuBLASLt products, the gated residual fused with
// the next norm, implicit-GEMM convolutions). Either computes each
// operation as the pinned PyTorch code rounds it (ops.h); kFast's products
// and fused norm write the same bits as kLegacy's, its convolutions sum in
// another order (conv.cu).
//
// Host inputs (token ids, rotary tables, the timestep's sinusoid and dt,
// the latents' statistics) are the caller's to upload to their places
// before a phase is queued; nothing here reads host memory or waits.

#ifndef LLMP_KERNELS_IMAGE_PIPELINE_H_
#define LLMP_KERNELS_IMAGE_PIPELINE_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "base/sha256.h"
#include "execution/registry.h"
#include "kernels/image/implementations.h"
#include "kernels/image/ops.h"
#include "model/qwen_image.h"

struct cublasContext;

namespace llmp::kernels::image {

class LtGemm;

// The pipeline's roles, a plan's operations in this order.
enum class Role : std::uint8_t {
  kTextEmbed,
  kTextNorm,
  kTextLinear,
  kTextHeadNormRope,
  kTextAttention,
  kTextAdd,
  kTextSwiGlu,
  kDitLinear,
  kDitSilu,
  kDitGelu,
  kDitTextNorm,
  kDitNorm,
  kDitHeadNormRope,
  kDitTextAttention,
  kDitAttention,
  kDitResidual,  // gated_residual, or gated_residual_norm (the next norm fused)
  kDitSwiGlu,
  kDitEuler,
  kVaeConvert,     // the F32 weights' BF16 copy (the paged runner's)
  kVaeConvert3x3,  // the 3x3 convolutions' weights: plain, or KRSC for the implicit GEMM
  kVaeTranspose,
  kVaeScaleShift,
  kVaeConv3x3,
  kVaeConv1x1,
  kVaeNorm,
  kVaeAdd,
  kVaeUpsample,
  kVaeDupUp,
  kVaeAttention,
  kVaeSoftMax,
  kCount
};
inline constexpr std::size_t kRoles = static_cast<std::size_t>(Role::kCount);
std::string_view RoleName(Role role);  // "text.embed", "dit.linear", ...

enum class PlanKind : std::uint8_t { kLegacy, kFast };
// The named plan's choices, in role order; `overrides` replaces a role's
// implementation by name ("dit.linear=image.linear.cublas").
std::expected<std::vector<execution::Choice>, std::string> QwenImageChoices(
    PlanKind kind, std::span<const std::string> overrides = {});

// The cuBLAS handles a plan's products use: `blas` (llmpalooza's, on the
// stream) for the cublasGemmEx implementations, `lt` for
// image.linear.cublaslt.
struct Handles {
  cublasContext* blas = nullptr;
  LtGemm* lt = nullptr;
};

// Device addresses (bytes). A phase's buffers in the shared workspace, and
// what lives from one step to the next in the image's own memory.
struct TextMemory {
  std::span<const std::uint64_t> tensors;  // TextEncoderTensors' order
  std::uint64_t x = 0, n = 0, q = 0, k = 0, v = 0, attn = 0, o = 0, g = 0, u = 0;
  std::uint64_t ids = 0;           // int32 [rows]
  std::uint64_t bad = 0;           // int32 flag, zeroed by the caller
  std::uint64_t cos = 0, sin = 0;  // BF16 [rows, 128]
  std::uint64_t embeds = 0;        // the kept rows' destination
  std::int64_t rows = 0;
  std::int64_t drop = 0;
};
struct DitMemory {
  std::span<const std::uint64_t> tensors;  // DenoiserTensors' order
  // The workspace.
  std::uint64_t x = 0, n = 0, q = 0, k = 0, v = 0, attn = 0, o = 0, g = 0, u = 0;
  std::uint64_t sin = 0;  // BF16 [2, timestep_dim], the step's sinusoid (uploaded)
  std::uint64_t dt = 0;   // F32, the step's dt rounded to BF16 (uploaded; EulerStepDt)
  std::uint64_t t1 = 0, t2 = 0, temb = 0, mod = 0, nscale = 0;
  // The image's own memory.
  std::uint64_t embeds = 0, txt = 0, pk = 0, pv = 0, freqs = 0, latents = 0, noise = 0;
  std::int64_t text = 0;
  std::int64_t image = 0;
};
struct VaeMemory {
  std::span<const std::uint64_t> f32;      // the artifact's F32 tensors, or empty
  std::span<const std::uint64_t> weights;  // BF16, as VaeWeightBf16 lays them out
  std::array<std::uint64_t, model::kVaeBuffers> buf{};
  std::uint64_t stats = 0;  // BF16 [2, 64]: std then mean (uploaded)
  std::uint64_t col = 0, col_bytes = 0;
  std::uint64_t scores = 0, scores_bytes = 0, probs = 0;
  std::uint64_t latents = 0;  // BF16 [grid * grid, z_dim]
  std::uint32_t grid = 0;
};

// Layouts: 256-byte aligned buffers from `base`, `bytes` their extent (base
// 0 to measure).
TextMemory TextLayout(std::uint64_t base, const model::QwenImageTextProfile& p, std::int64_t rows,
                      std::uint64_t& bytes);
DitMemory DitLayout(std::uint64_t base, const model::QwenImageDenoiserProfile& p, std::int64_t rows,
                    std::uint64_t& bytes);
// The image's own memory for `text` prompt rows and `image` image rows.
DitMemory OwnLayout(std::uint64_t base, const model::QwenImageDenoiserProfile& p, std::int64_t text,
                    std::int64_t image, std::uint64_t& bytes);
// The VAE's working memory; with `bf16_weights`, the weights' BF16 copies
// first (their F32 sizes `f32_bytes`), filling `weights`.
VaeMemory VaeLayout(std::uint64_t base, const model::QwenImageVaeProfile& p, std::uint32_t grid,
                    std::span<const std::uint64_t> f32_bytes, bool bf16_weights,
                    std::vector<std::uint64_t>& weights, std::uint64_t& bytes);
// The im2col buffer's size, which sets how im2col splits a convolution and
// so its products' shapes (the M3 slice's).
inline constexpr std::uint64_t kIm2colBytes = std::uint64_t{256} << 20U;

class QwenImagePipeline {
 public:
  // Refused, with nothing bound, if the plan does not resolve against the
  // registry (execution::Resolve), a role's implementation is not this
  // module's (identity and all) or not one the role takes, or the VAE's
  // weight layout and its convolution disagree.
  static std::expected<std::unique_ptr<QwenImagePipeline>, std::string> Bind(
      const execution::Registry& registry, const execution::Plan& plan,
      const model::QwenImageProfile& profile);

  QwenImagePipeline(const QwenImagePipeline&) = delete;
  QwenImagePipeline& operator=(const QwenImagePipeline&) = delete;
  QwenImagePipeline(QwenImagePipeline&&) = delete;
  QwenImagePipeline& operator=(QwenImagePipeline&&) = delete;
  ~QwenImagePipeline();

  const base::Sha256Digest& identity() const { return identity_; }
  Impl bound(Role role) const { return bound_.at(static_cast<std::size_t>(role)); }
  // Each role and its implementation, JSON.
  std::string Describe() const;

  // The text encoder over m.rows tokens (ids, cos and sin at their
  // places), the rows [drop, rows) copied to m.embeds.
  Status Encode(const TextMemory& m, const Handles& h, Stream s) const;
  // txt_in: the text rows of the joint sequence (m.txt) from m.embeds.
  Status TextRows(const DitMemory& m, const Handles& h, Stream s) const;
  // One denoising step: the noise prediction (m.noise) for m.latents with
  // the sinusoid at m.sin, then the Euler update of m.latents by *m.dt. The
  // first step runs the joint sequence and fills the prefix K/V cache
  // (m.pk, m.pv); later ones only the image rows.
  Status Step(bool first, const DitMemory& m, const Handles& h, Stream s) const;
  // The VAE's BF16 weights from m.f32 into m.weights (the paged runner's).
  Status ConvertVae(const VaeMemory& m, Stream s) const;
  // The decoder: m.latents to the decoded [out_channels, 16 grid, 16 grid]
  // in m.buf[kVaeX].
  Status Decode(const VaeMemory& m, const Handles& h, Stream s) const;

  // A VAE tensor's BF16 copy as Decode reads it, on the host (the resident
  // harness's): its F32 bytes (little-endian) rounded to nearest even into
  // `out` (one value per float), a 3x3 convolution's weights in KRSC order
  // when the plan's convolution takes them so.
  Status VaeWeightBf16(std::size_t tensor, std::span<const std::byte> f32,
                       std::span<Bf16> out) const;

 private:
  QwenImagePipeline() = default;
  Status Linear(Role role, const Handles& h, std::uint64_t x, std::int64_t ldx, std::uint64_t w,
                std::int64_t ldw, std::uint64_t out, std::int64_t ldo, std::int64_t m,
                std::int64_t n, std::int64_t k, Stream s) const;
  Status Block(std::uint32_t b, bool first, const DitMemory& m, const Handles& h, Stream s) const;

  model::QwenImageProfile profile_;
  std::array<Impl, kRoles> bound_{};
  base::Sha256Digest identity_{};
  // Per VAE tensor: [co, ci] of a 3x3 convolution's weights, else {0, 0};
  // and its element count.
  std::vector<std::array<std::uint32_t, 2>> conv3x3_;
  std::vector<std::uint64_t> vae_elements_;
};

}  // namespace llmp::kernels::image

#endif  // LLMP_KERNELS_IMAGE_PIPELINE_H_
