// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/image/implementations.h"

#include <cublas_api.h>

#include <array>
#include <cstddef>
#include <expected>
#include <format>
#include <string>
#include <string_view>
#include <vector>

#include "execution/registry.h"

// The build's part of each identity, from CMakeLists.txt.
#if !defined(LLMP_IMAGE_SDK) || !defined(LLMP_IMAGE_TARGET) ||                    \
    !defined(LLMP_IMAGE_CUDA_ARCHITECTURES) || !defined(LLMP_IMAGE_BUILD_TYPE) || \
    !defined(LLMP_IMAGE_SANITIZE)
#error "implementations.cc needs the SDK, target, architectures and build type"
#endif

namespace llmp::kernels::image {
namespace {

using execution::Operation;

struct Entry {
  std::string_view name;
  Operation operation;
  Impl impl;
  bool cublas;  // its identity records the pinned cuBLAS
  std::string_view variant;
};

#ifdef NDEBUG
constexpr std::string_view kAsserts = "NDEBUG";
#else
constexpr std::string_view kAsserts = "asserts";
#endif
// And whether it checks libstdc++'s preconditions (D-083), as the GGML and
// EXL3 modules' identities record.
#ifdef _GLIBCXX_ASSERTIONS
constexpr std::string_view kLibraryAsserts = "libstdc++ assertions";
#else
constexpr std::string_view kLibraryAsserts = "no libstdc++ assertions";
#endif

constexpr std::array<Entry, static_cast<std::size_t>(Impl::kCount)> kEntries = {{
    {"image.embed_rows", Operation::kGetRows, Impl::kEmbedRows, false,
     "EmbedRowsKernel: one block of 256 threads per row, ids checked on the device"},
    {"image.rms_norm", Operation::kRmsNormMul, Impl::kRmsNorm, false,
     "RmsNormKernel: w * bf16(x * rsqrt(mean(x^2) + eps)), one block of 512 per row"},
    {"image.rms_norm.zero_centre", Operation::kRmsNormMul, Impl::kRmsNormZeroCentre, false,
     "ZeroCenterRmsNormKernel: bf16(x * rsqrt(mean(x^2) + eps) * (w + 1)) in F32"},
    {"image.linear.cublas", Operation::kMatMul, Impl::kLinearCublas, true,
     "cublasGemmEx (T, N), BF16 operands and output, COMPUTE_32F, "
     "CUBLAS_GEMM_DEFAULT_TENSOR_OP, the handle's workspace"},
    {"image.linear.cublaslt", Operation::kMatMul, Impl::kLinearCublasLt, true,
     "cublasLtMatmul (T, N), BF16 operands and output, COMPUTE_32F, alpha 1, beta 0: the "
     "algorithm pinned for the shape by gemm.cc's table on its device and cuBLASLt, else "
     "cuBLASLt's first heuristic choice for the workspace; operands 256-byte aligned"},
    {"image.head_norm_rope.neox", Operation::kHeadNormRope, Impl::kHeadNormRopeNeox, false,
     "HeadNormRopeNeoxKernel: a warp per row and head of 128, rotate_half, BF16 tables"},
    {"image.head_norm_rope.complex", Operation::kHeadNormRope, Impl::kHeadNormRopeComplex, false,
     "HeadNormRopeComplexKernel<BF16, in place>: a warp per row and head of 128, F32 complex "
     "frequencies"},
    {"image.attention.short", Operation::kFlashAttn, Impl::kAttentionShort, false,
     "SmallAttentionKernel: one block of 128 per query row and head, F32 scores and softmax "
     "in shared memory, at most 1,024 rows"},
    {"image.flash_attention", Operation::kFlashAttn, Impl::kFlashAttention, false,
     "FlashForwardKernel: FlashAttention-2 forward, 128 query rows and 8 warps per block, "
     "64-key tiles two stages deep, mma.sync m16n8k16 BF16, F32 online softmax"},
    {"image.add", Operation::kAdd, Impl::kAdd, false, "AddKernel: BF16 + BF16, one rounding"},
    {"image.swiglu", Operation::kSwiGlu, Impl::kSwiGlu, false,
     "SwiGlu8Kernel (8 elements per thread) where widths and strides are multiples of 8, "
     "else SwiGluKernel: bf16(bf16(silu(gate)) * up)"},
    {"image.silu", Operation::kUnary, Impl::kSilu, false, "SiluKernel: silu in F32, rounded"},
    {"image.gelu_tanh", Operation::kUnary, Impl::kGeluTanh, false,
     "GeluTanhKernel: PyTorch's tanh GELU in F32, rounded"},
    {"image.layer_norm_modulate", Operation::kLayerNormModulate, Impl::kLayerNormModulate, false,
     "LayerNormModulateKernel<chunks, false>: one block of 256 per row, the chunks in "
     "registers"},
    {"image.gated_residual", Operation::kGatedResidual, Impl::kGatedResidual, false,
     "GatedResidualKernel: 8 elements per thread"},
    {"image.gated_residual_norm", Operation::kGatedResidualNorm, Impl::kGatedResidualNorm, false,
     "LayerNormModulateKernel<chunks, true>: the gated residual written back, then its "
     "layer_norm_modulate, one pass, LayerNormModulate's sums"},
    {"image.euler_step", Operation::kEulerStep, Impl::kEulerStep, false,
     "EulerStepAtKernel: dt (BF16-rounded) read from the device"},
    {"image.transpose", Operation::kCont, Impl::kTranspose, false, "TransposeKernel"},
    {"image.scale_shift", Operation::kScale, Impl::kScaleShift, false,
     "ScaleShiftChannelsKernel: bf16(bf16(z * std) + mean)"},
    {"image.conv2d.im2col", Operation::kConv2d, Impl::kConvIm2col, true,
     "3x3: FillBiasKernel, then per block of whole rows fitting 256 MiB Im2Col3x3Kernel and "
     "cublasGemmEx (N, N, beta 1)"},
    {"image.conv2d.implicit", Operation::kConv2d, Impl::kConvImplicit, false,
     "3x3: Conv3x3Kernel, implicit GEMM over KRSC weights, 128 pixels x 144 channels per "
     "block, 16 input channels per stage, two stages, mma.sync m16n8k16 BF16, F32, the bias "
     "added before one rounding"},
    {"image.conv2d.product", Operation::kConv2d, Impl::kConvProduct, true,
     "1x1: FillBiasKernel, then cublasGemmEx (N, N, beta 1)"},
    {"image.channel_norm", Operation::kChannelNorm, Impl::kChannelNorm, false,
     "ChannelRmsNormKernel: a thread per pixel, then SiLU where the step asks"},
    {"image.upsample", Operation::kUpsample, Impl::kUpsample, false, "Upsample2xKernel"},
    {"image.dup_up_add", Operation::kDupUpAdd, Impl::kDupUpAdd, false, "AddDupUpKernel"},
    {"image.matmul.single_head", Operation::kMatMul, Impl::kMatmulSingleHead, true,
     "cublasGemmEx (N, T) into F32 scores and (T, N) into BF16, COMPUTE_32F, "
     "CUBLAS_GEMM_DEFAULT_TENSOR_OP"},
    {"image.soft_max", Operation::kSoftMax, Impl::kSoftMax, false,
     "SoftmaxRowsKernel: one block of 512 per row, F32, BF16 out"},
    {"image.convert", Operation::kConvert, Impl::kConvert, false,
     "F32ToBf16Kernel: round to nearest even"},
    {"image.convert.krsc", Operation::kConvert, Impl::kConvertKrsc, false,
     "KrscFromF32Kernel: [co, ci, 3, 3] F32 to [co, 3, 3, ci] BF16, round to nearest even"},
    {"image.flash_attention.prefixed", Operation::kFlashAttn, Impl::kFlashAttentionPrefixed, false,
     "FlashForwardKernel as image.flash_attention, the keys and values below the prefix read "
     "from the prefix cache in place of a copy ahead of the call"},
    {"image.flash_attention.norm_q", Operation::kFlashAttn, Impl::kFlashAttentionNormQ, false,
     "FlashForwardKernel<norm q> as image.flash_attention.prefixed, each query row normed and "
     "rotated in registers as HeadNormRopeComplexKernel computes it (its sums' order), past the "
     "first step; the first step as image.flash_attention"},
}};

execution::Implementation Declaration(const Entry& entry) {
  std::string build = std::format(
      "sdk {}; target {}; cuda architectures {}; build type {} ({}, {}); sanitize {}; module "
      "sources {}",
      LLMP_IMAGE_SDK, LLMP_IMAGE_TARGET, LLMP_IMAGE_CUDA_ARCHITECTURES, LLMP_IMAGE_BUILD_TYPE,
      kAsserts, kLibraryAsserts, LLMP_IMAGE_SANITIZE, ModuleSourcesDigest());
  if (entry.cublas) {
    build += std::format("; cuBLAS {}", CUBLAS_VERSION);
  }
  return execution::Implementation{.name = std::string(entry.name),
                                   .operation = entry.operation,
                                   .source = "llmp",
                                   .revision = std::string(ModuleSourcesDigest()),
                                   .build = std::move(build),
                                   .variant = std::string(entry.variant)};
}

}  // namespace

std::vector<execution::Implementation> Implementations() {
  std::vector<execution::Implementation> declared;
  declared.reserve(kEntries.size());
  for (const Entry& entry : kEntries) {
    declared.push_back(Declaration(entry));
  }
  return declared;
}

std::expected<Impl, std::string> Bind(const execution::Implementation& implementation) {
  for (const Entry& entry : kEntries) {
    if (entry.name != implementation.name) {
      continue;
    }
    if (execution::IdentityOf(Declaration(entry)) != execution::IdentityOf(implementation)) {
      return std::unexpected(
          std::format("{} is not this build's {}", implementation.name, entry.name));
    }
    return entry.impl;
  }
  return std::unexpected(
      std::format("the image module has no implementation {}", implementation.name));
}

std::string_view ImplName(Impl impl) {
  const auto i = static_cast<std::size_t>(impl);
  return i < kEntries.size() ? kEntries.at(i).name : std::string_view("unknown");
}

}  // namespace llmp::kernels::image
