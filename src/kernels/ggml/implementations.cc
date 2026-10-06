// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/implementations.h"

#include <array>
#include <cstddef>
#include <expected>
#include <format>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "execution/registry.h"
#include "ggml.h"
#include "kernels/ggml/dsv4_hc_norm.h"
#include "kernels/ggml/dsv4_outa.h"
#include "kernels/ggml/dsv4_qhead.h"
#include "kernels/ggml/dsv4_weighted_reduce.h"
#include "kernels/ggml/fattn_owner.h"
#include "kernels/ggml/gemma_moe_fusion.h"
#include "kernels/ggml/gemma_norm.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/ops_ext.h"
#include "kernels/ggml/tensors.h"
#include "kernels/ggml/validate.h"
#include "kernels/ggml/validate_ext.h"

// The build's part of each identity, from CMakeLists.txt.
#if !defined(JITLLM_GGML_SOURCE_TREE) || !defined(JITLLM_DS4_SOURCE_TREE) ||        \
    !defined(JITLLM_GGML_SDK) || !defined(JITLLM_GGML_TARGET) ||                    \
    !defined(JITLLM_GGML_CUDA_ARCHITECTURES) || !defined(JITLLM_GGML_BUILD_TYPE) || \
    !defined(JITLLM_GGML_SANITIZE)
#error "implementations.cc needs the GGML source tree, SDK, target, architectures and build type"
#endif

namespace jitllm::kernels::ggml {

struct RmsNormMulKernel::Entry {
  std::string_view name;
  std::string_view variant;
  std::expected<void, KernelFailure> (*check)(const ggml_tensor* norm, const ggml_tensor* mul);
  std::expected<void, KernelFailure> (*run)(LaunchContext& launch, ggml_tensor* norm,
                                            ggml_tensor* mul);
};

struct Kernel::Entry {
  std::string_view name;
  execution::Operation operation;
  std::string_view variant;
  std::size_t arity;
  // Called with exactly `arity` nodes.
  std::expected<void, KernelFailure> (*check)(std::span<const ggml_tensor* const> nodes);
  std::expected<void, KernelFailure> (*run)(LaunchContext& launch,
                                            std::span<ggml_tensor* const> nodes);
};

namespace {

// Whether this build keeps asserts, GGML's device asserts among them.
#ifdef NDEBUG
constexpr std::string_view kAsserts = "NDEBUG";
#else
constexpr std::string_view kAsserts = "asserts";
#endif
// And whether it checks libstdc++'s preconditions (D-083).
#ifdef _GLIBCXX_ASSERTIONS
constexpr std::string_view kLibraryAsserts = "libstdc++ assertions";
#else
constexpr std::string_view kLibraryAsserts = "no libstdc++ assertions";
#endif

constexpr std::array<RmsNormMulKernel::Entry, 2> kRmsNormMul = {{
    {.name = "ggml.rms_norm_mul.fused",
     .variant = "ggml_cuda_op_rms_norm_fused; upstream launch configuration",
     .check = &CheckRmsNormMul,
     .run = &RmsNormMul},
    {.name = "ggml.rms_norm_mul.unfused",
     .variant = "ggml_cuda_op_rms_norm, then ggml_cuda_op_mul; upstream launch configuration",
     .check = &CheckRmsNormThenMul,
     .run = &RmsNormThenMul},
}};

// The other implementations (implementations.h), each checking and running
// its nodes through ops.h.
using Nodes = std::span<ggml_tensor* const>;
using ConstNodes = std::span<const ggml_tensor* const>;

// Matching only reads descriptors. Borrow a fixed stack view for the shared
// graph matcher; no tensor metadata or ownership is changed here.
template <std::size_t N>
std::array<ggml_tensor*, N> Borrow(ConstNodes nodes) {
  std::array<ggml_tensor*, N> out{};
  for (std::size_t i = 0; i < N; ++i) out[i] = const_cast<ggml_tensor*>(nodes[i]);
  return out;
}
std::unexpected<KernelFailure> InvalidGemmaChain() {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = "invalid checked Gemma MoE chain"});
}
constexpr std::array<Kernel::Entry, 122> kKernels = {{
    {.name = kGemmaRouteName,
     .operation = execution::Operation::kGemmaRoute,
     .variant = "original ggml_cuda_op_topk_moe; Gemma128/top8/clamp2^-14; "
                "both outputs, full sort backing retained",
     .arity = 10,
     .check = [](ConstNodes n) -> std::expected<void, KernelFailure> {
       const auto nodes = Borrow<10>(n);
       return GemmaRoutingFusionAt(nodes, 0) ? std::expected<void, KernelFailure>{}
                                             : InvalidGemmaChain();
     },
     .run = [](LaunchContext& l, Nodes n) -> std::expected<void, KernelFailure> {
       const auto f = GemmaRoutingFusionAt(n, 0);
       if (!f) return InvalidGemmaChain();
       return RunGemmaRouting(l, f->operands);
     }},
    {.name = kGemmaReduceName,
     .operation = execution::Operation::kGemmaScaledReduce,
     .variant = "original ggml_cuda_op_moe_weighted_reduction; "
                "F32[2816,8,rows], (expert*scale)*weight, ascending selected slots",
     .arity = 17,
     .check = [](ConstNodes n) -> std::expected<void, KernelFailure> {
       const auto nodes = Borrow<17>(n);
       return GemmaReductionFusionAt(nodes, 0) ? std::expected<void, KernelFailure>{}
                                               : InvalidGemmaChain();
     },
     .run = [](LaunchContext& l, Nodes n) -> std::expected<void, KernelFailure> {
       const auto f = GemmaReductionFusionAt(n, 0);
       if (!f) return InvalidGemmaChain();
       return RunGemmaScaledReduction(l, f->operands);
     }},
    {.name = kGemmaNormRopeName,
     .operation = execution::Operation::kRmsNormMulRope,
     .variant = "original ggml_cuda_op_rms_norm_mul_rope_fused; F32 full D256/D512 NEOX, "
                "no direct cache store; default-off checked three-node chain",
     .arity = 3,
     .check = [](ConstNodes n) { return CheckGemmaNormRope(n[0], n[1], n[2]); },
     .run = [](LaunchContext& l, Nodes n) { return RunGemmaNormRope(l, n[0], n[1], n[2]); }},
    {.name = kGemmaNormAddName,
     .operation = execution::Operation::kRmsNormMulAdd,
     .variant = "original ggml_cuda_op_rms_norm_fused_add; F32 approved Gemma widths, "
                "default-off checked three-node chain",
     .arity = 3,
     .check = [](ConstNodes n) { return CheckGemmaNormAdd(n[0], n[1], n[2]); },
     .run = [](LaunchContext& l, Nodes n) { return RunGemmaNormAdd(l, n[0], n[1], n[2]); }},
    {.name = "ggml.rms_norm",
     .operation = execution::Operation::kRmsNorm,
     .variant = "ggml_cuda_op_rms_norm: rms_norm_f32<block, false, false>; upstream launch "
                "configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckRmsNorm(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RmsNorm(launch, n[0]); }},
    {.name = kDsv4QHeadName,
     .operation = execution::Operation::kRope,
     .variant = "QHeadKernel: native 256-thread RMS reduction at width512, explicit F32 "
                "normalization rounding and normal tail64 RoPE multiply/FMA order",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckDsv4QHead(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunDsv4QHead(launch, n[0]); }},
    {.name = kDsv4F16CopyName,
     .operation = execution::Operation::kConvert,
     .variant = "F16CopyKernel: RN F32-to-F16 copy shared by F16-weight cuBLAS products "
                "(experimental)",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckDsv4F16Copy(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunDsv4F16Copy(launch, n[0]); }},
    {.name = kDsv4HcNormF16Name,
     .operation = execution::Operation::kRmsNorm,
     .variant = "HcNormF16Kernel: native 1024-thread flat RMS reduction and F32 scale, then RN "
                "F16 rows (or the F32 rows) for the HC mix product (experimental)",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckDsv4HcNormF16(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunDsv4HcNormF16(launch, n[0]); }},
    {.name = kDsv4HcPostExpertsNormF16Name,
     .operation = execution::Operation::kHcPost,
     .variant = "HcPostNormF16Kernel<true>: the ordered six-slot expert reduction and shared add "
                "formed in the HC post, then F32 streams and F16 (or F32) mix-input rows "
                "(experimental)",
     .arity = 4,
     .check = [](ConstNodes n) { return CheckDsv4HcPostExpertsNormF16(n[0], n[1], n[2], n[3]); },
     .run = [](LaunchContext& launch,
               Nodes n) { return RunDsv4HcPostExpertsNormF16(launch, n[0], n[1], n[2], n[3]); }},
    {.name = kDsv4HcPostNormF16Name,
     .operation = execution::Operation::kHcPost,
     .variant = "HcPostNormF16Kernel: native HC post FMAs into F32 streams, then the native "
                "flat RMS and RN F16 (or F32) mix-input rows (experimental)",
     .arity = 2,
     .check = [](ConstNodes n) { return CheckDsv4HcPostNormF16(n[0], n[1]); },
     .run = [](LaunchContext& launch,
               Nodes n) { return RunDsv4HcPostNormF16(launch, n[0], n[1]); }},
    {.name = kDsv4OutAName,
     .operation = execution::Operation::kMatMul,
     .variant = "paid raw-Q8 packing, inverse tail64 RoPE, staged F16/HMMA grouped output-A "
                "into canonical F32 output; borrowed ds4 numerical core",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckDsv4OutA(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunDsv4OutA(launch, n[0]); }},
    {.name = kDsv4OutAFastPackName,
     .operation = execution::Operation::kMatMul,
     .variant = "output-A with a coalesced raw-Q8 weight repack (identical bytes), then the same "
                "staged F16/HMMA grouped product (experimental)",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckDsv4OutA(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunDsv4OutA(launch, n[0], true); }},
    {.name = "ggml.add",
     .operation = execution::Operation::kAdd,
     .variant = "ggml_cuda_op_add: k_bin_bcast<op_add, float, float, float>; upstream launch "
                "configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckBinary(n[0], GGML_OP_ADD); },
     .run = [](LaunchContext& launch, Nodes n) { return Add(launch, n[0]); }},
    {.name = "ggml.mul",
     .operation = execution::Operation::kMul,
     .variant = "ggml_cuda_op_mul: k_bin_bcast<op_mul, float, float, float>; upstream launch "
                "configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckBinary(n[0], GGML_OP_MUL); },
     .run = [](LaunchContext& launch, Nodes n) { return Mul(launch, n[0]); }},
    {.name = "ggml.mul_mat.mmvf",
     .operation = execution::Operation::kMatMul,
     .variant = "ggml_cuda_mul_mat_vec_f: mul_mat_vec_f<T, type_acc, ncols, block, false, false> "
                "as upstream selects; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMulMat(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return MulMatVecF(launch, n[0]); }},
    {.name = "ggml.mul_mat.mmf",
     .operation = execution::Operation::kMatMul,
     .variant = "ggml_cuda_mul_mat_f: mul_mat_f<T, warp, cols, nwarps, false> as upstream "
                "selects; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMulMatF(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return MulMatF(launch, n[0]); }},
    {.name = "ggml.mul_mat.cublas",
     .operation = execution::Operation::kMatMul,
     .variant = "GGML's cuBLAS path (mul_mat_cublas.cu): conversions, GemmEx, strided or "
                "pointer-array batched GEMM on the lent handle, as upstream plans them",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMulMatCublasOperands(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return MulMatCublas(launch, n[0]); }},
    {.name = "ggml.get_rows",
     .operation = execution::Operation::kGetRows,
     .variant =
         "ggml_cuda_op_get_rows: F32 rows through k_get_rows_float_vec on 16-byte "
         "vectors, aligned rows and at least 128 blocks, else k_get_rows_float; F16 and BF16 rows "
         "through k_get_rows_float<half or nv_bfloat16, float>; I32 rows keep their type; "
         "upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckGetRows(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return GetRows(launch, n[0]); }},
    {.name = "ggml.set_rows",
     .operation = execution::Operation::kSetRows,
     .variant = "ggml_cuda_op_set_rows: k_set_rows<float, int64_t, half>; upstream launch "
                "configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckSetRows(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return SetRows(launch, n[0]); }},
    {.name = "ggml.rope.neox",
     .operation = execution::Operation::kRope,
     .variant = "ggml_cuda_op_rope: rope_neox<true, false, float, float>; upstream launch "
                "configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckRope(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return Rope(launch, n[0]); }},
    {.name = "ggml.rope_set_rows.fused",
     .operation = execution::Operation::kRopeSetRows,
     .variant = "ggml_cuda_op_rope_fused: rope_neox<true, false, float, half> writing the KV "
                "destination; upstream launch configuration",
     .arity = 2,
     .check = [](ConstNodes n) { return CheckRopeSetRows(n[0], n[1]); },
     .run = [](LaunchContext& launch, Nodes n) { return RopeSetRows(launch, n[0], n[1]); }},
    {.name = "ggml.soft_max",
     .operation = execution::Operation::kSoftMax,
     .variant = "ggml_cuda_op_soft_max: soft_max_f32<true, ncols, block, float> for 32 to 4,096 "
                "columns in powers of two, else soft_max_f32<true, 0, 0, float>, rows in shared "
                "memory only; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckSoftMax(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return SoftMax(launch, n[0]); }},
    {.name = "ggml.cont",
     .operation = execution::Operation::kCont,
     .variant = "ggml_cuda_dup: cudaMemcpyAsync if contiguous, cudaMemcpy2DAsync for a pitched "
                "block, else cpy_scalar<cpy_1_scalar<float, float>>; no tiled transpose; upstream "
                "launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) -> std::expected<void, KernelFailure> {
       if (auto checked = CheckCont(n[0]); !checked) {
         return std::unexpected(checked.error());
       }
       return {};
     },
     .run = [](LaunchContext& launch, Nodes n) { return Cont(launch, n[0]); }},
    {.name = "ggml.swiglu",
     .operation = execution::Operation::kSwiGlu,
     .variant = "ggml_cuda_op_swiglu: unary_gated_op_kernel<op_silu, float>; upstream launch "
                "configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckSwiGlu(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return SwiGlu(launch, n[0]); }},
    {.name = "ggml.geglu",
     .operation = execution::Operation::kGeGlu,
     .variant = "ggml_cuda_op_geglu: unary_gated_op_kernel<op_gelu, float>; upstream launch "
                "configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckGeGlu(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return GeGlu(launch, n[0]); }},
    {.name = "ggml.convert",
     .operation = execution::Operation::kConvert,
     .variant = "ggml_cuda_cpy between packed tensors: cpy_scalar_contiguous<float, half> or "
                "<half, float>; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckConvert(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return Convert(launch, n[0]); }},
    {.name = "ggml.flash_attn_ext.vec",
     .operation = execution::Operation::kFlashAttn,
     .variant = "ggml_cuda_flash_attn_ext_vec_case<64, F16, F16>, forced: "
                "flash_attn_mask_to_KV_max<ncols> from 1,024 query rows, flash_attn_ext_vec<64, "
                "1 or 2, F16, F16, false>, flash_attn_combine_results<64> over parallel blocks; "
                "launch_fattn's launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckFlashAttnVec(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return FlashAttnVec(launch, n[0]); }},
    {.name = "ggml.mul_mat_add.mmvf_fused",
     .operation = execution::Operation::kMulMatAdd,
     .variant = "ggml_cuda_mul_mat_vec_f with x_bias, writing the add: mul_mat_vec_f<T, "
                "type_acc, 1, block, true, false>, the add's precision; upstream launch "
                "configuration",
     .arity = 2,
     .check = [](ConstNodes n) { return CheckMulMatVecBias(n[0], n[1]); },
     .run = [](LaunchContext& launch, Nodes n) { return MulMatVecBias(launch, n[0], n[1]); }},
    {.name = "ggml.mul_mat_glu.mmvf_fused",
     .operation = execution::Operation::kMulMatGlu,
     .variant = "ggml_cuda_mul_mat_vec_f with gate and SwiGLU, writing the GLU: "
                "mul_mat_vec_f<T, type_acc, 1, block, true, false>, the GLU's parameters as "
                "precision; upstream launch configuration",
     .arity = 3,
     .check = [](ConstNodes n) { return CheckMulMatVecGlu(n[0], n[1], n[2]); },
     .run = [](LaunchContext& launch, Nodes n) { return MulMatVecGlu(launch, n[0], n[1], n[2]); }},
    {.name = "ggml.mul_mat_geglu.mmvf_fused",
     .operation = execution::Operation::kMulMatGeGlu,
     .variant = "ggml_cuda_mul_mat_vec_f with gate and GELU-tanh GLU, writing the GLU: "
                "mul_mat_vec_f<T, type_acc, 1, block, true, false>, the GLU's parameters as "
                "precision; upstream launch configuration",
     .arity = 3,
     .check = [](ConstNodes n) { return CheckMulMatVecGeGlu(n[0], n[1], n[2]); },
     .run = [](LaunchContext& launch,
               Nodes n) { return MulMatVecGeGlu(launch, n[0], n[1], n[2]); }},
    // DeepSeek V4 Flash and Qwen3.8 Flash (ops_ext.h).
    {.name = "ggml.mul_mat.mmvq",
     .operation = execution::Operation::kMatMul,
     .variant = "ggml_cuda_mul_mat_vec_q: quantize_row_q8_1_cuda, then mul_mat_vec_q<type, "
                "ncols_dst> as upstream selects; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMulMatQ(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return MulMatVecQ(launch, n[0]); }},
    {.name = "ggml.mul_mat.mmq",
     .operation = execution::Operation::kMatMul,
     .variant = "ggml_cuda_mul_mat_q: quantize_mmq_q8_1_cuda (or the native FP4 quantization for "
                "MXFP4 on Blackwell), then mul_mat_q<type, J, fallback> with J and stream-k as "
                "upstream selects, and its fixup; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMulMatQ(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return MulMatQ(launch, n[0]); }},
    {.name = "ggml.mul_mat.fwht",
     .operation = execution::Operation::kMatMul,
     .variant = "ggml_cuda_op_fwht for GGML_HINT_SRC0_IS_HADAMARD: fwht_cuda<n>, the weights "
                "never read; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMulMatHadamard(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return MulMatHadamard(launch, n[0]); }},
    {.name = "ggml.mul_mat_id.mmvq",
     .operation = execution::Operation::kMulMatId,
     .variant =
         "ggml_cuda_mul_mat_vec_q with ids: quantize_row_q8_1_cuda, then mul_mat_vec_q<type, "
         "tokens> over each token's selected experts; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMulMatIdQ(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return MulMatVecQ(launch, n[0]); }},
    {.name = "ggml.mul_mat_id.mmq",
     .operation = execution::Operation::kMulMatId,
     .variant = "ggml_cuda_mul_mat_q with ids: ggml_cuda_launch_mm_ids_helper, the activations "
                "quantized (or scattered) per selected expert, then mul_mat_q<type, J, fallback> "
                "over each expert's tokens and its fixup; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMulMatIdQ(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return MulMatQ(launch, n[0]); }},
    {.name = "jitllm.mul_mat_id.mmq_pair",
     .operation = execution::Operation::kMulMatId,
     .variant = "two ordinary MMQ expert products sharing one routing map and type-specific "
                "Q8 preparation, preserving each weight/output stride and sequential fixup",
     .arity = 2,
     .check = [](ConstNodes n) { return CheckMulMatIdQPair(n[0], n[1]); },
     .run = [](LaunchContext& launch, Nodes n) { return MulMatIdQPair(launch, n[0], n[1]); }},
    {.name = "jitllm.mul_mat_id.mmq_compact",
     .operation = execution::Operation::kMulMatId,
     .variant = "ordinary non-FP4 MMQ preparation and inner product with a device-built "
                "expert-major tile list; original weight/output strides and Q8 arithmetic",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMulMatIdQCompact(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return MulMatIdQCompact(launch, n[0]); }},
    {.name = "jitllm.mul_mat_id.q2_d2r",
     .operation = execution::Operation::kMulMatId,
     .variant = "ds4 down_q2k_d2r_kernel<64,64,raw>: 128x64 tiles; native maps/Q8 D2S6; raw Q2_K",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMulMatIdQ2D2r(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return MulMatIdQ2D2r(launch, n[0]); }},
    {.name = "jitllm.mul_mat_id.mmq_pair_compact",
     .operation = execution::Operation::kMulMatId,
     .variant = "two ordinary non-FP4 MMQ inner products sharing routing and Q8 preparation, "
                "each launched with a device-built expert-major tile list",
     .arity = 2,
     .check = [](ConstNodes n) { return CheckMulMatIdQPair(n[0], n[1]); },
     .run = [](LaunchContext& launch, Nodes n) { return MulMatIdQPair(launch, n[0], n[1], true); }},
    {.name = "jitllm.mul_mat.mmq_pair_dense",
     .operation = execution::Operation::kMatMul,
     .variant = "two dense MMQ products of one block-quantized non-FP4 type sharing one Q8_1 "
                "quantization of their activation (experimental)",
     .arity = 2,
     .check = [](ConstNodes n) { return CheckMulMatQPairDense(n[0], n[1]); },
     .run = [](LaunchContext& launch, Nodes n) { return MulMatQPairDense(launch, n[0], n[1]); }},
    {.name = "jitllm.mul_mat_id.mmq_pair_glu",
     .operation = execution::Operation::kMulMatId,
     .variant = "GB10 IQ2 occupancy-two compact pair, gate first, up write-back storing "
                "swiglu_clamp(gate, up) (experimental)",
     .arity = 3,
     .check = [](ConstNodes n) { return CheckMulMatIdQPairGlu(n[0], n[1], n[2]); },
     .run = [](LaunchContext& launch,
               Nodes n) { return MulMatIdQPairGlu(launch, n[0], n[1], n[2]); }},
    {.name = "jitllm.mul_mat_id.mmq_pair_glu_q8",
     .operation = execution::Operation::kMulMatId,
     .variant = "the activation write-back pair, its activation stored as the down product's "
                "D2S6 Q8_1 MMQ input at sorted columns (experimental)",
     .arity = 3,
     .check = [](ConstNodes n) { return CheckMulMatIdQPairGlu(n[0], n[1], n[2]); },
     .run = [](LaunchContext& launch,
               Nodes n) { return MulMatIdQPairGluQ8(launch, n[0], n[1], n[2]); }},
    {.name = "jitllm.mul_mat_id.mmq_compact_prequant",
     .operation = execution::Operation::kMulMatId,
     .variant = "compact Q2_K MMQ over the pair write-back's D2S6 input, unquantized here "
                "(experimental)",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMulMatIdQCompactPrequant(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return MulMatIdQCompactPrequant(launch, n[0]); }},
    {.name = "ggml.sub",
     .operation = execution::Operation::kSub,
     .variant = "ggml_cuda_op_sub: k_bin_bcast<op_sub, float, float, float>; upstream launch "
                "configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckBinary(n[0], GGML_OP_SUB); },
     .run = [](LaunchContext& launch, Nodes n) { return Sub(launch, n[0]); }},
    {.name = "ggml.div",
     .operation = execution::Operation::kDiv,
     .variant = "ggml_cuda_op_div: k_bin_bcast<op_div, float, float, float>; upstream launch "
                "configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckBinary(n[0], GGML_OP_DIV); },
     .run = [](LaunchContext& launch, Nodes n) { return Div(launch, n[0]); }},
    {.name = "ggml.scale",
     .operation = execution::Operation::kScale,
     .variant = "ggml_cuda_op_scale: scale_f32; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckScale(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return Scale(launch, n[0]); }},
    {.name = "ggml.unary",
     .operation = execution::Operation::kUnary,
     .variant =
         "ggml_cuda_op_<function> for abs, sgn, neg, silu, GELU-tanh, tanh, relu, sigmoid, exp, "
         "softplus and sqrt: unary_op_kernel<op_<function>, float>; upstream launch "
         "configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckUnary(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return Unary(launch, n[0]); }},
    {.name = "ggml.clamp",
     .operation = execution::Operation::kClamp,
     .variant = "ggml_cuda_op_clamp: op_clamp_kernel<float>; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckClamp(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return Clamp(launch, n[0]); }},
    {.name = "ggml.fill",
     .operation = execution::Operation::kFill,
     .variant = "ggml_cuda_op_fill: fill_kernel<float or half>; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckFill(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return Fill(launch, n[0]); }},
    {.name = "ggml.repeat",
     .operation = execution::Operation::kRepeat,
     .variant = "ggml_cuda_op_repeat: k_bin_bcast<op_repeat, float, float, float>; upstream "
                "launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckRepeat(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return Repeat(launch, n[0]); }},
    {.name = "ggml.concat",
     .operation = execution::Operation::kConcat,
     .variant = "ggml_cuda_op_concat: concat_cont<T, dim> per sample for contiguous operands, two "
                "copies along the samples, else concat_non_cont<T, dim>; upstream launch "
                "configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckConcat(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return Concat(launch, n[0]); }},
    {.name = "ggml.sum_rows",
     .operation = execution::Operation::kSumRows,
     .variant = "ggml_cuda_op_sum_rows: reduce_rows_f32<false>, 512 threads per row below two "
                "rows per multiprocessor, else 32 or 128; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckSumRows(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return SumRows(launch, n[0]); }},
    {.name = "ggml.argsort.bitonic",
     .operation = execution::Operation::kArgsort,
     .variant = "ggml_cuda_op_argsort for rows of at most 1,024 that fit shared memory: "
                "k_argsort_f32_i32<order>; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckArgsort(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return Argsort(launch, n[0]); }},
    {.name = "ggml.top_k.radix",
     .operation = execution::Operation::kTopK,
     .variant = "ggml_cuda_op_top_k without CUB: top_k_radix_cuda (8-bit radix select, unordered) "
                "for rows over 1,024, else k_argsort_f32_i32<DESC> and a pitched copy of the first "
                "k; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckTopK(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return TopK(launch, n[0]); }},
    {.name = "ggml.swiglu_clamp",
     .operation = execution::Operation::kSwiGluClamp,
     .variant = "ggml_cuda_op_swiglu_clamp: swiglu_clamp_kernel<float>; upstream launch "
                "configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckSwiGluClamp(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return SwiGluClamp(launch, n[0]); }},
    {.name = "ggml.rope.ext",
     .operation = execution::Operation::kRope,
     .variant = "ggml_cuda_op_rope or ggml_cuda_op_rope_back: rope_norm, rope_neox or rope_multi "
                "<forward, false, float, float> by the node's mode, with its offset and YaRN; "
                "upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckRopeExt(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RopeExt(launch, n[0]); }},
    {.name = "ggml.get_rows.ext",
     .operation = execution::Operation::kGetRows,
     .variant = "ggml_cuda_op_get_rows: k_get_rows<qk, qr, dequantize> for Q8_0, k_get_rows_kq<"
                "dequantize_type> for the k- and i-quants and MXFP4, k_get_rows_float<int32_t, "
                "int32_t> for I32; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckGetRowsExt(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return GetRowsExt(launch, n[0]); }},
    {.name = "ggml.set_rows.ext",
     .operation = execution::Operation::kSetRows,
     .variant = "ggml_cuda_op_set_rows: k_set_rows<src, I32 or I64, dst> for F32 into F32 or F16 "
                "and F16 into F16; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckSetRowsExt(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return SetRowsExt(launch, n[0]); }},
    {.name = "ggml.ssm_conv",
     .operation = execution::Operation::kSsmConv,
     .variant = "ggml_cuda_op_ssm_conv unfused: ssm_conv_f32<false, 128, conv> up to 32 tokens, "
                "else ssm_conv_long_token_f32<false, 128, conv, 32>; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckSsmConv(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return SsmConv(launch, n[0]); }},
    {.name = "ggml.gated_delta_net",
     .operation = execution::Operation::kGatedDeltaNet,
     .variant = "ggml_cuda_op_gated_delta_net: gated_delta_net_cuda<S, KDA, snapshots>; upstream "
                "launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckGatedDeltaNet(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return GatedDeltaNet(launch, n[0]); }},
    {.name = "ggml.lightning_indexer.wmma",
     .operation = execution::Operation::kLightningIndexer,
     .variant = "ggml_cuda_lightning_indexer on tensor cores: lightning_indexer_kernel_wmma<8, 32, "
                "128, heads, F16>; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckLightningIndexer(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return LightningIndexer(launch, n[0]); }},
    {.name = "ggml.dsv4_hc_comb",
     .operation = execution::Operation::kHcComb,
     .variant = "ggml_cuda_op_dsv4_hc_comb: dsv4_hc_comb_f32; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckHcComb(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return HcComb(launch, n[0]); }},
    {.name = "ggml.dsv4_hc_pre",
     .operation = execution::Operation::kHcPre,
     .variant = "ggml_cuda_op_dsv4_hc_pre: dsv4_hc_pre_f32; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckHcPre(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return HcPre(launch, n[0]); }},
    {.name = "ggml.dsv4_hc_post",
     .operation = execution::Operation::kHcPost,
     .variant = "ggml_cuda_op_dsv4_hc_post: dsv4_hc_post_f32; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckHcPost(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return HcPost(launch, n[0]); }},
    {.name = "ggml.flash_attn_ext.vec_d256",
     .operation = execution::Operation::kFlashAttn,
     .variant = "pinned D256/F16 vector attention, exact GQA2; parallel split and original fixup",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckFlashAttnVec256(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return FlashAttnVec256(launch, n[0]); }},
    {.name = "ggml.flash_attn_ext.mma_gqa2",
     .operation = execution::Operation::kFlashAttn,
     .variant = "pinned D256/group2 MMA, query tiles4/8/16/32; dense mask and stream-k fixup",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckFlashAttnMmaGqa2(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return FlashAttnMmaGqa2(launch, n[0]); }},
    {.name = "ggml.flash_attn_ext.mma",
     .operation = execution::Operation::kFlashAttn,
     .variant = "ggml_cuda_flash_attn_ext_mma_f16_case<D, D, 1, 2, 4 or 8, 8> for D 256 and 512 as "
                "switch_ncols1 picks, the sparse gather at D 512 as upstream decides, launch_fattn "
                "with stream-k and its fixup; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckFlashAttnMma(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return FlashAttnMma(launch, n[0]); }},
    {.name = "jitllm.dsv4.hca_tokentile",
     .operation = execution::Operation::kFlashAttn,
     .variant = "ds4 attention_tokentile_hmma_kernel: four tokens/G8, M32/R32, 16 warps; "
                "F32 Q rounded to F16, original F32 QK/softmax/PV accumulation and F16 "
                "probabilities; native F16 ring mirror and original dense causal records",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckDsv4HcaTokentile(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return Dsv4HcaTokentile(launch, n[0]); }},
    {.name = "jitllm.flash_attn_ext.mma_wide",
     .operation = execution::Operation::kFlashAttn,
     .variant = "explicit sparse query-union choice at D256/512, one or eight query columns; "
                "original per-query masks and F16 KV, stream-k with fixup",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckFlashAttnMma(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return FlashAttnMma(launch, n[0], true); }},
    {.name = "ggml.flash_attn_ext.mma_d128",
     .operation = execution::Operation::kFlashAttn,
     .variant = "ggml_cuda_flash_attn_ext_mma_f16_case<128, 128, 8, 16, 32 or 64, 1> as "
                "switch_ncols1 picks, no mask, launch_fattn with stream-k and its fixup; upstream "
                "launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckFlashAttnMma128(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return FlashAttnMma128(launch, n[0]); }},
    // jitLLM's own (jitllm_ops.h), for Qwen3.8's MXFP8 and NVFP4 tensors.
    {.name = "jitllm.mxfp8.mul_mat_vec",
     .operation = execution::Operation::kMatMul,
     .variant = "Mxfp8Gemv<columns 1-8, rows 1/4/2 a warp>: each row's sums as one warp a row's, "
                "16-code vectors, F32 block sums; PDL, the first weights prefetched into L2",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMxfp8MulMatVec(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunMxfp8MulMatVec(launch, n[0]); }},
    {.name = "jitllm.mxfp8.dequant",
     .operation = execution::Operation::kConvert,
     .variant = "Mxfp8ToBf16: sixteen codes a thread",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMxfp8Dequant(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunMxfp8Dequant(launch, n[0]); }},
    {.name = "jitllm.nvfp4.get_rows",
     .operation = execution::Operation::kGetRows,
     .variant = "Nvfp4RowsKernel: a block an id, a thread a value",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckNvfp4Rows(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunNvfp4Rows(launch, n[0]); }},
    {.name = "jitllm.qrows.get_rows",
     .operation = execution::Operation::kGetRows,
     .variant = "QRowsKernel<type>: a block an id, a thread a value, GGML's dequantize.cuh",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckQRows(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunQRows(launch, n[0]); }},
    // jitLLM's fusions of Qwen3.8's GGML nodes (jitllm_ops.h), GGML's
    // arithmetic in its order.
    {.name = "jitllm.hc.combine",
     .operation = execution::Operation::kHcCombine,
     .variant = "HcCombineKernel: four columns a thread",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckHcCombine(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunHcCombine(launch, n[0]); }},
    {.name = "jitllm.hc.norm",
     .operation = execution::Operation::kHcNorm,
     .variant = "HcNormKernel<F32 or BF16>: a stream of a token a 1,024-thread block, "
                "rms_norm_f32<1024>'s reduction",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckHcNorm(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunHcNorm(launch, n[0]); }},
    {.name = "jitllm.hc.mix",
     .operation = execution::Operation::kHcMix,
     .variant = "HcMixKernel: a token a 1,024-thread block, rms_norm_f32<1024>'s reduction per "
                "stream",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckHcMix(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunHcMix(launch, n[0]); }},
    {.name = "jitllm.moe.glu",
     .operation = execution::Operation::kMoeGlu,
     .variant = "MoeGluKernel: four columns a thread",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMoeGlu(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunMoeGlu(launch, n[0]); }},
    {.name = "jitllm.moe.combine",
     .operation = execution::Operation::kMoeCombine,
     .variant = "MoeCombineKernel: four columns a thread, experts in order",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMoeCombine(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunMoeCombine(launch, n[0]); }},
    {.name = "jitllm.bf16",
     .operation = execution::Operation::kConvert,
     .variant = "Bf16Kernel: __float2bfloat16, an element a thread",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckBf16(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunBf16(launch, n[0]); }},
    {.name = "jitllm.gemm.bf16",
     .operation = execution::Operation::kMatMul,
     .variant =
         "cublasGemmEx BF16 x BF16 into F32, CUBLAS_COMPUTE_32F, default tensor-op algorithm; "
         "GemvBf16 nodes of one column GemvBf16Rows or GemvBf16Split (PDL, F32 sums)",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckGemmBf16(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunGemmBf16(launch, n[0]); }},
    {.name = "jitllm.gated_delta_net.columns",
     .operation = execution::Operation::kGatedDeltaNet,
     .variant = "GdnColumnsKernel<4>: gated_delta_net_cuda<128>'s per-column arithmetic, four "
                "value columns a warp, four warps a block",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckGatedDeltaNetColumns(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunGatedDeltaNetColumns(launch, n[0]); }},
    {.name = "jitllm.gated_delta_net.lanes",
     .operation = execution::Operation::kGatedDeltaNet,
     .variant = "GdnLanesKernel: a value column over 8 lanes of 16 rows, 64 columns a block, "
                "16-token chunks staged in shared memory by asynchronous copies",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckGatedDeltaNetLanes(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunGatedDeltaNetLanes(launch, n[0]); }},
    {.name = "jitllm.gdn.conv",
     .operation = execution::Operation::kSsmConv,
     .variant = "GdnConvKernel<F32 or BF16 rows>: a head of a token a 128-thread block; "
                "ssm_conv, silu, and rms_norm_f32<256>'s reduction for the query and key heads",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckGdnConv(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunGdnConv(launch, n[0]); }},
    {.name = "jitllm.gdn.norm_gate",
     .operation = execution::Operation::kNormGate,
     .variant = "GdnNormGateKernel<F32 or BF16 out, F32 or BF16 z>: a head a warp, "
                "rms_norm_f32<256>'s reduction; GdnNormGateMxfp8Kernel<z> into MXFP8 rows",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckGdnNormGate(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunGdnNormGate(launch, n[0]); }},
    // The routed experts over the CUTLASS layout (jitllm_ops.h, moe_layout.h).
    {.name = "jitllm.moe.route",
     .operation = execution::Operation::kMoeRoute,
     .variant = "RouteCount, RouteScan, RouteAssign: 64-token chunks, slots in token order",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMoeRoute(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunMoeRoute(launch, n[0]); }},
    {.name = "jitllm.moe.quantize",
     .operation = execution::Operation::kQuantize,
     .variant = "QuantizeRows: quantize_mmq_nvfp4's row and block scales, a token a block",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMoeQuantize(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunMoeQuantize(launch, n[0]); }},
    {.name = "jitllm.moe.gemm.cutlass",
     .operation = execution::Operation::kMulMatId,
     .variant = "CUTLASS 4.7.1 Sm120 block-scaled NVFP4 grouped GEMM, KernelPtrArrayTmaWarp"
                "SpecializedPingpong, tile 128x128x256, cluster 1x1x1, BF16 LinearCombination "
                "epilogue; device-side problem shapes from SetupGroups",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMoeGemm(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunMoeGemm(launch, n[0]); }},
    {.name = "jitllm.moe.glu_quantize",
     .operation = execution::Operation::kMoeGlu,
     .variant = "GluQuantizeRows: SwiGLU then quantize_mmq_nvfp4's scales, a row a block",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMoeGluQuantize(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunMoeGluQuantize(launch, n[0]); }},
    {.name = "jitllm.moe.combine_sorted",
     .operation = execution::Operation::kMoeCombine,
     .variant = "CombineSorted: four columns a thread, experts in order",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMoeCombineSorted(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunMoeCombineSorted(launch, n[0]); }},
    {.name = "jitllm.moe.gemv",
     .operation = execution::Operation::kMulMatId,
     .variant = "Gemv: a warp an output row of a slot, 16-value blocks a lane, F32 activations; "
                "PDL",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMoeGemv(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunMoeGemv(launch, n[0]); }},
    // A speculative verify's row-invariant products (D-092) and DeepSeek's
    // DSpark drafter's argmax.
    {.name = "jitllm.mul_mat.mmvq_rows",
     .operation = execution::Operation::kMatMul,
     .variant = "MulMatVecQRowsKernel<type, columns 1-8>: quantize_row_q8_1_cuda, then GGML's "
                "mul_mat_vec_q body with the one-column launch's warps, rows per block, small-K "
                "and halved iterations for every column; GGML's device flags",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMulMatQ(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return MulMatVecQRows(launch, n[0]); }},
    {.name = "jitllm.mul_mat_id.mmvq_rows",
     .operation = execution::Operation::kMulMatId,
     .variant = "MulMatVecQRowsKernel<type, 1, per token>: quantize_row_q8_1_cuda, then GGML's "
                "one-token mul_mat_vec_q launch with ids for every token, over grid z",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMulMatIdQ(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return MulMatVecQRows(launch, n[0]); }},
    {.name = "jitllm.mul_mat.mmvf_rows",
     .operation = execution::Operation::kMatMul,
     .variant = "ggml_cuda_mul_mat_vec_f: mul_mat_vec_f<T, type_acc, columns 1-8, block, false, "
                "false> whatever upstream would route; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMulMat(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return MulMatVecFRows(launch, n[0]); }},
    {.name = "jitllm.argmax",
     .operation = execution::Operation::kTopK,
     .variant = "ArgmaxKernel<probability>: a block a row, the highest value, the lowest index "
                "among equals; optionally its softmax probability, a second pass in a fixed "
                "order",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckArgmax(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunArgmax(launch, n[0]); }},
    // Qwen3.8's fast path (jitllm_ops.h): the MXFP8 products on tensor cores.
    {.name = "jitllm.mxfp8.quantize",
     .operation = execution::Operation::kQuantize,
     .variant = "Mxfp8QuantizeKernel<F32 or BF16>: a 32-value block a thread, E8M0 scale "
                "2^ceil(log2(amax / 448)), E4M3 codes rounded to nearest, swizzled scales",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMxfp8Quantize(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunMxfp8Quantize(launch, n[0]); }},
    {.name = "jitllm.mxfp8.swizzle",
     .operation = execution::Operation::kConvert,
     .variant = "Mxfp8SwizzleKernel: a scale a thread, rows padded to 128",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMxfp8Swizzle(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunMxfp8Swizzle(launch, n[0]); }},
    {.name = "jitllm.mxfp8.gemm.cutlass",
     .operation = execution::Operation::kMatMul,
     .variant = "CUTLASS 4.7.1 Sm120 block-scaled MXFP8 GEMM, KernelTmaWarpSpecializedPingpong, "
                "tile 128x128x128, cluster 1x1x1, F32 or BF16 LinearCombination epilogue, tiles "
                "swizzled 8 along N past 4,096 rows",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMxfp8Gemm(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunMxfp8Gemm(launch, n[0]); }},
    {.name = "jitllm.hc.prep",
     .operation = execution::Operation::kHcNorm,
     .variant = "HcPrepKernel<combine, inject>: a token a block, a float4 column of every stream "
                "a thread; the combine, the streams' RMS norms into BF16, the inject logits. Up to "
                "8 tokens, on compute capability 9.0 and later, HcPrepClusterKernel: a token a "
                "cluster of 8 blocks, the sums through distributed shared memory in block order",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckHcPrep(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunHcPrep(launch, n[0]); }},
    {.name = "jitllm.hc.lo",
     .operation = execution::Operation::kUnary,
     .variant = "HcLoKernel: silu(lo / hc) into BF16, an element a thread",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckHcLo(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunHcLo(launch, n[0]); }},
    {.name = "jitllm.hc.mix_bf16",
     .operation = execution::Operation::kHcMix,
     .variant = "HcMixBf16Kernel: eight columns of a token a thread, BF16 streams and logits",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckHcMixBf16(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunHcMixBf16(launch, n[0]); }},
    {.name = "jitllm.moe.router",
     .operation = execution::Operation::kArgsort,
     .variant = "MoeRouterKernel: a token a warp; softmax, top experts by warp argmax, their "
                "weights renormalized, the shared expert's gate logit",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMoeRouter(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunMoeRouter(launch, n[0]); }},
    {.name = "jitllm.gdn.history",
     .operation = execution::Operation::kCont,
     .variant = "GdnHistoryKernel<F32 or BF16>: a channel a thread, the last rows transposed, "
                "after the old history's last taps where the rows are fewer",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckGdnHistory(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunGdnHistory(launch, n[0]); }},
    {.name = "jitllm.gdn.gates",
     .operation = execution::Operation::kUnary,
     .variant = "GdnGatesKernel: paired sigmoid(beta) and softplus(alpha + dt_bias) * ssm_a, "
                "the original GGML F32 rounding points",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckGdnGates(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunGdnGates(launch, n[0]); }},
    {.name = "jitllm.gdn.step",
     .operation = execution::Operation::kGatedDeltaNet,
     .variant = "GdnColumnsKernel<4> over the state in place: the columns kernel's arithmetic, "
                "the new state written over the old, the attention output alone",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckGdnStep(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunGdnStep(launch, n[0]); }},
    {.name = "jitllm.qsa.prep",
     .operation = execution::Operation::kRope,
     .variant = "QsaPrepKernel: a head of a token a warp; rms_norm times the weight, then "
                "rope_multi's NEOX pairs at the token's position over 64 dimensions",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckQsaPrep(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunQsaPrep(launch, n[0]); }},
    {.name = "jitllm.qsa.gate_quantize",
     .operation = execution::Operation::kQuantize,
     .variant = "QsaGateQuantizeKernel: a 32-value block a thread, attention times "
                "sigmoid(gate), MXFP8",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckQsaGateQuantize(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunQsaGateQuantize(launch, n[0]); }},
    {.name = "jitllm.qsa.pool",
     .operation = execution::Operation::kRope,
     .variant = "QsaPoolKernel: a block the chunk completes a warp; its raw keys summed in cell "
                "order over the ratio, then qsa.prep's norm and rotation at its first position, "
                "BF16 into the block keys in place",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckQsaPool(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunQsaPool(launch, n[0]); }},
    {.name = "jitllm.qsa.topk",
     .operation = execution::Operation::kTopK,
     .variant = "QsaQueryBf16Kernel, then QsaScoreVecKernel (a block a thread, up to 16 tokens) "
                "or QsaScoreMmaKernel (BF16 m16n8k16, 128 block keys in registers, 16-token "
                "tiles), keys in scratch; QsaSelectTileKernel (a token's 8,192-block tile a "
                "256-thread block, four-pass radix select over per-warp histograms, ties "
                "to the lower cell) and past "
                "one tile QsaSelectMergeKernel over the tiles' candidates",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckQsaTopK(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunQsaTopK(launch, n[0]); }},
    {.name = "jitllm.qsa.attn",
     .operation = execution::Operation::kFlashAttn,
     .variant = "QsaAttnKernel: a warp a token's KV head and share of its cells, 16-cell "
                "cp.async gathers of K and V double-buffered, F16 m16n8k16 with F32 sums and "
                "online softmax; QsaAttnCombineKernel (a query head a block) over the shares in "
                "order",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckQsaAttn(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunQsaAttn(launch, n[0]); }},
    // DeepSeek V4's fast plan (jitllm_ops.h; dsv4_fast.cu).
    {.name = "jitllm.q8_1",
     .operation = execution::Operation::kQuantize,
     .variant = "quantize_row_q8_1_cuda: rows padded to 512 values, once for every product",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckQuantizeQ8(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunQuantizeQ8(launch, n[0]); }},
    {.name = "jitllm.vecq",
     .operation = execution::Operation::kMatMul,
     .variant = "VecQKernel<type, rows, warps, glu>: GGML's vec_dot_*_q8_1 over up to 8 tokens a "
                "weight read, one block per distinct expert and row block; SwiGLU in the kernel",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckVecQ(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunVecQ(launch, n[0]); }},
    {.name = "jitllm.dsv4.route",
     .operation = execution::Operation::kMoeRoute,
     .variant = "RouteKernel: a warp a token, sqrt(softplus), top-k by argmax rounds, normalized",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckDsv4Route(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunDsv4Route(launch, n[0]); }},
    {.name = "jitllm.dsv4.combine",
     .operation = execution::Operation::kMoeCombine,
     .variant = "CombineKernel: four columns a thread, experts in order, then the shared expert",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckDsv4Combine(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunDsv4Combine(launch, n[0]); }},
    {.name = kDsv4WeightedReduceName,
     .operation = execution::Operation::kMoeCombine,
     .variant = "WeightedReduceKernel: one column per thread, initial rounded F32 multiply "
                "then five ascending rounded multiply/adds; no contraction or scratch",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckDsv4OrderedReduce(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunDsv4OrderedReduce(launch, n[0]); }},
    {.name = "jitllm.dsv4.hc_mix",
     .operation = execution::Operation::kHcMix,
     .variant = "HcMixKernel: 64 chunks of a token, 24 dot products and the sum of squares",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckDsv4HcMix(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunDsv4HcMix(launch, n[0]); }},
    {.name = "jitllm.dsv4.hc_pre",
     .operation = execution::Operation::kHcPre,
     .variant = "HcPreKernel: a 1,024-thread block a token; mixes, Sinkhorn, weighted sum, "
                "RMSNorm times the weight",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckDsv4HcPre(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunDsv4HcPre(launch, n[0]); }},
    {.name = "jitllm.dsv4.compress",
     .operation = execution::Operation::kSoftMax,
     .variant = "CompressKernel: a block a compressed block, a thread a channel; the rows' "
                "online softmax and weighted sum",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckDsv4Compress(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunDsv4Compress(launch, n[0]); }},
    {.name = "jitllm.dsv4.lid_topk",
     .operation = execution::Operation::kLightningIndexer,
     .variant = "LidScoreKernel<R>: R rows' 64 heads a block on mma.sync F16 (F32 sums), 64 keys "
                "a step; TopKKernel: a block a row, a radix select, ties to the lower row",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckDsv4LidTopK(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunDsv4LidTopK(launch, n[0]); }},
    {.name = kFlashAttnOwnersName,
     .operation = execution::Operation::kFlashAttn,
     .variant = "closed C4 F16 independent K/V roots; original packed MMA grid and partitions",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckFlashAttnOwnersNode(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) -> std::expected<void, KernelFailure> {
       auto inputs = FlashAttnOwnersFromNode(n[0]);
       if (!inputs) return std::unexpected(inputs.error());
       return FlashAttnOwnerRoots(launch, *inputs);
     }},
    {.name = kGemma4MaskName,
     .operation = execution::Operation::kFill,
     .variant = "packed F16 global causal/local ring mask from fresh segment I32 positions",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckGemma4Mask(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunGemma4Mask(launch, n[0]); }},
    {.name = "jitllm.dsv4.sparse_mask",
     .operation = execution::Operation::kFill,
     .variant = "SparseMaskKernel: a block a row, the window's mask copied, -inf, then the "
                "selected rows 0",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckDsv4SparseMask(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunDsv4SparseMask(launch, n[0]); }},
}};

execution::Implementation Declare(std::string_view name, execution::Operation operation,
                                  std::string_view variant) {
  // The grouped GEMM and the MXFP8 product are CUTLASS's kernels: their
  // identities name that tree too.
  const bool cutlass = name == kMoeGemmName || name == kMxfp8GemmName;
  const bool ds4 = name == kMulMatIdQ2D2r || name == kDsv4HcaTokentileName ||
                   name == kDsv4OutAName || name == kDsv4OutAFastPackName;
  std::string source = "ggml";
  std::string revision =
      std::format("ggml tree {}; jitllm module {}", JITLLM_GGML_SOURCE_TREE, ModuleSourcesDigest());
  if (ds4) {
    source = "ds4";
    revision = std::format("ds4 tree {}; ggml tree {}; jitllm module {}", JITLLM_DS4_SOURCE_TREE,
                           JITLLM_GGML_SOURCE_TREE, ModuleSourcesDigest());
  } else if (cutlass) {
    source = "cutlass";
    revision =
        std::format("cutlass tree {}; ggml tree {}; jitllm module {}", JITLLM_CUTLASS_SOURCE_TREE,
                    JITLLM_GGML_SOURCE_TREE, ModuleSourcesDigest());
  }
  return {
      .name = std::string(name),
      .operation = operation,
      .source = std::move(source),
      .revision = std::move(revision),
      .build = std::format("sdk {}; target {}; cuda {}; build type {}; {}; {}; sanitizers {}",
                           JITLLM_GGML_SDK, JITLLM_GGML_TARGET, JITLLM_GGML_CUDA_ARCHITECTURES,
                           JITLLM_GGML_BUILD_TYPE, kAsserts, kLibraryAsserts, JITLLM_GGML_SANITIZE),
      .variant = std::string(variant)};
}

execution::Implementation Declare(const RmsNormMulKernel::Entry& entry) {
  return Declare(entry.name, execution::Operation::kRmsNormMul, entry.variant);
}

execution::Implementation Declare(const Kernel::Entry& entry) {
  return Declare(entry.name, entry.operation, entry.variant);
}

}  // namespace

std::vector<execution::Implementation> Implementations() {
  std::vector<execution::Implementation> declared;
  declared.reserve(kRmsNormMul.size() + kKernels.size());
  for (const RmsNormMulKernel::Entry& entry : kRmsNormMul) {
    declared.push_back(Declare(entry));
  }
  for (const Kernel::Entry& entry : kKernels) {
    declared.push_back(Declare(entry));
  }
  return declared;
}

std::expected<RmsNormMulKernel, KernelFailure> RmsNormMulKernel::Bind(
    const execution::Implementation& implementation) {
  for (const Entry& entry : kRmsNormMul) {
    if (entry.name != implementation.name) {
      continue;
    }
    if (execution::IdentityOf(Declare(entry)) != execution::IdentityOf(implementation)) {
      break;
    }
    return RmsNormMulKernel(entry);
  }
  return std::unexpected(KernelFailure{
      .error = KernelError::kRejected,
      .detail = std::format("{} is not a GGML rms_norm_mul implementation of this build",
                            implementation.name)});
}

std::expected<void, KernelFailure> RmsNormMulKernel::Check(const ggml_tensor* norm,
                                                           const ggml_tensor* mul) const {
  return entry_->check(norm, mul);
}

std::expected<void, KernelFailure> RmsNormMulKernel::Run(LaunchContext& launch, ggml_tensor* norm,
                                                         ggml_tensor* mul) const {
  return entry_->run(launch, norm, mul);
}

std::string_view RmsNormMulKernel::name() const { return entry_->name; }

bool UsesCublas(std::string_view implementation) {
  // The declarations above whose launchers take the lent cuBLAS handle
  // (MulMatCublas, RunGemmBf16); a new one belongs here.
  return implementation == "ggml.mul_mat.cublas" || implementation == "jitllm.gemm.bf16";
}

std::expected<Kernel, KernelFailure> Kernel::Bind(const execution::Implementation& implementation) {
  for (const Entry& entry : kKernels) {
    if (entry.name != implementation.name) {
      continue;
    }
    if (execution::IdentityOf(Declare(entry)) != execution::IdentityOf(implementation)) {
      break;
    }
    return Kernel(entry);
  }
  return std::unexpected(KernelFailure{
      .error = KernelError::kRejected,
      .detail = std::format("{} is not a GGML implementation of this build", implementation.name)});
}

std::expected<void, KernelFailure> Kernel::Check(std::span<const ggml_tensor* const> nodes) const {
  if (nodes.size() != entry_->arity) {
    return std::unexpected(KernelFailure{
        .error = KernelError::kRejected,
        .detail =
            std::format("{} takes {} nodes, not {}", entry_->name, entry_->arity, nodes.size())});
  }
  return entry_->check(nodes);
}

std::expected<void, KernelFailure> Kernel::Run(LaunchContext& launch,
                                               std::span<ggml_tensor* const> nodes) const {
  if (nodes.size() != entry_->arity) {
    return std::unexpected(KernelFailure{
        .error = KernelError::kRejected,
        .detail =
            std::format("{} takes {} nodes, not {}", entry_->name, entry_->arity, nodes.size())});
  }
  return entry_->run(launch, nodes);
}

std::string_view Kernel::name() const { return entry_->name; }

execution::Operation Kernel::operation() const { return entry_->operation; }

std::size_t Kernel::arity() const { return entry_->arity; }

}  // namespace jitllm::kernels::ggml
