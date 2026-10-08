// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The image module's entries in the implementation registry (D-053;
// execution/registry.h), CUDA builds only: a build without the device code
// declares none, so a plan naming one is unsupported there (BP-S4).
//
//   image.embed_rows              get_rows             ops.h EmbedRows
//   image.rms_norm                rms_norm_mul         ops.h RmsNorm (Qwen3-VL's)
//   image.rms_norm.zero_centre    rms_norm_mul         ops.h ZeroCenterRmsNorm
//   image.linear.cublas           mul_mat              gemm.h Linear (cublasGemmEx)
//   image.linear.cublaslt         mul_mat              gemm.h LtGemm (pinned algorithms)
//   image.head_norm_rope.neox     head_norm_rope       ops.h HeadNormRopeNeox
//   image.head_norm_rope.complex  head_norm_rope       ops.h HeadNormRopeComplex
//   image.attention.short         flash_attn           ops.h SmallAttention
//   image.flash_attention         flash_attn           ops.h FlashAttention
//   image.add                     add                  ops.h Add
//   image.swiglu                  swiglu               ops.h SwiGlu
//   image.silu                    unary                ops.h Silu
//   image.gelu_tanh               unary                ops.h GeluTanh
//   image.layer_norm_modulate     layer_norm_modulate  ops.h LayerNormModulate
//   image.gated_residual          gated_residual       ops.h GatedResidual
//   image.gated_residual_norm     gated_residual_norm  ops.h GatedResidualNorm
//   image.euler_step              euler_step           ops.h EulerStepAt
//   image.transpose               cont                 ops.h Transpose
//   image.scale_shift             scale                ops.h ScaleShiftChannels
//   image.conv2d.im2col           conv2d               ops.h FillBias, Im2Col3x3 and
//                                                      gemm.h ConvProduct (3x3)
//   image.conv2d.implicit         conv2d               ops.h Conv3x3Implicit (3x3)
//   image.conv2d.product          conv2d               ops.h FillBias, gemm.h
//                                                      ConvProduct (1x1)
//   image.channel_norm            channel_norm         ops.h ChannelRmsNorm
//   image.upsample                upsample             ops.h Upsample2x
//   image.dup_up_add              dup_up_add           ops.h AddDupUp
//   image.matmul.single_head      mul_mat              gemm.h ScoresQtK and
//                                                      ValuesTimesProbs
//   image.soft_max                soft_max             ops.h SoftmaxRowsToBf16
//   image.convert                 convert              ops.h F32ToBf16
//   image.convert.krsc            convert              ops.h Conv3x3WeightsKrsc
//   image.flash_attention.prefixed flash_attn          ops.h FlashAttentionPrefixed
//                                                      (the prefix cache read in
//                                                      place, not copied first)
//   image.flash_attention.norm_q  flash_attn           the same, the queries normed
//                                                      and rotated in registers
//                                                      (ops.h QueryNorm) in place of
//                                                      HeadNormRopeComplex on q
//
// Each identity covers everything that decides what an implementation
// computes and launches: llmpalooza's code in the module (a digest of every
// file in src/kernels/image, written at build time, module_digest.cmake,
// so an edit here, the pinned algorithms' table included, changes every
// identity the module declares); the SDK, target, device architectures,
// build type and sanitizers; for the cuBLAS paths the pinned cuBLAS; the
// name and a variant naming the kernels.

#ifndef LLMP_KERNELS_IMAGE_IMPLEMENTATIONS_H_
#define LLMP_KERNELS_IMAGE_IMPLEMENTATIONS_H_

#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

#include "execution/registry.h"

namespace llmp::kernels::image {

// What this module declares to the registry.
std::vector<execution::Implementation> Implementations();

// The digest of the module's own files, generated at build time.
std::string_view ModuleSourcesDigest();

// The module's implementations, as a bound plan names them.
enum class Impl : std::uint8_t {
  kEmbedRows,
  kRmsNorm,
  kRmsNormZeroCentre,
  kLinearCublas,
  kLinearCublasLt,
  kHeadNormRopeNeox,
  kHeadNormRopeComplex,
  kAttentionShort,
  kFlashAttention,
  kAdd,
  kSwiGlu,
  kSilu,
  kGeluTanh,
  kLayerNormModulate,
  kGatedResidual,
  kGatedResidualNorm,
  kEulerStep,
  kTranspose,
  kScaleShift,
  kConvIm2col,
  kConvImplicit,
  kConvProduct,
  kChannelNorm,
  kUpsample,
  kDupUpAdd,
  kMatmulSingleHead,
  kSoftMax,
  kConvert,
  kConvertKrsc,
  kFlashAttentionPrefixed,
  kFlashAttentionNormQ,
  kCount
};

// The implementation `implementation` declares, refused unless it is one
// this module declares, identity and all: a stale or foreign declaration
// never selects a kernel.
std::expected<Impl, std::string> Bind(const execution::Implementation& implementation);

std::string_view ImplName(Impl impl);

}  // namespace llmp::kernels::image

#endif  // LLMP_KERNELS_IMAGE_IMPLEMENTATIONS_H_
