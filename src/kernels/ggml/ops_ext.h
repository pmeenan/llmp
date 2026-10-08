// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The GGML-derived implementations of the operations DeepSeek V4 Flash
// (GGUF) and Qwen3.8 Flash add beyond the backend proof's (ops.h), over the
// K-C launch context (D-053). As there, each checks every precondition its
// GGML launcher asserts (validate_ext.h), so an unsupported operand is a
// rejection and never an abort, and queues upstream's kernels on the
// context's stream. An operation that draws scratch from the context's pool
// computes its bound first, as upstream's launcher will draw it, so that
// the pool's bound is never exceeded (launch.h).
//
// The quantized matrix products come as GGML's two kernel families, vector
// (MMVQ) and tile (MMQ), which the plan selects between as upstream's
// routing would (SelectMulMatQ); flash attention comes as the tensor-core
// (MMA) kernels for head dimensions 256 and 512, grouped 8 query heads to a
// KV head, as upstream's MMA dispatch groups them on the GB10
// (fattn.cu:218-268). Upstream would take its vector kernel instead for a
// D = 256 decode row over fewer than 8,192 cells (fattn.cu:610-618), which
// is not built; the MMA kernel runs those rows here.
//
// Row indices and expert ids read from device memory are the plan's to
// bound (validate_ext.h).

#ifndef JITLLM_KERNELS_GGML_OPS_EXT_H_
#define JITLLM_KERNELS_GGML_OPS_EXT_H_

#include <cstdint>
#include <expected>
#include <span>

#include "ggml.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/tensors.h"
#include "kernels/ggml/validate_ext.h"

namespace jitllm::kernels::ggml {

// Quantized matrix products: a ggml_mul_mat or ggml_mul_mat_id node with
// weights of a compiled quantized type (validate_ext.h CheckMulMatQ,
// CheckMulMatIdQ, QuantMulMatPath).
// What upstream's routing chooses on the context's device
// (ggml-cuda.cu:1864-1871 for mul_mat, 1924-1942 for mul_mat_id), refused
// where it would take neither family.
std::expected<QuantMulMatPath, KernelFailure> SelectMulMatQ(const LaunchContext& launch,
                                                            const ggml_tensor* node);
// Original dense device/type/column selector without requiring bound operands.
// Final planning still authenticates operands and the original MMVQ footprint.
bool DenseMmvqShapeSelected(const LaunchContext& launch, ggml_type type, std::int64_t columns);
// The pool scratch each family's launcher draws for a node it takes, each
// block from a 256-byte boundary in the launcher's allocation order: MMVQ's
// Q8_1 activations (mmvq.cu:1484-1486); MMQ's expert maps for mul_mat_id,
// its quantized activations and the stream-k fixup buffer
// (mmq.cu:199-278, mmq.cuh:1446-1455).
std::expected<std::uint64_t, KernelFailure> PlanMulMatVecQ(const LaunchContext& launch,
                                                           const ggml_tensor* node);
std::expected<std::uint64_t, KernelFailure> PlanMulMatQ(const LaunchContext& launch,
                                                        const ggml_tensor* node);
// Launches the family on a node upstream routes to it: ggml_cuda_mul_mat_vec_q
// or ggml_cuda_mul_mat_q, with the node's ids for mul_mat_id.
std::expected<void, KernelFailure> MulMatVecQ(LaunchContext& launch, ggml_tensor* node);
// Same ordinary MMVQ consumer, with its original padded Q8_1 draw supplied
// as a graph activation. Validates the original device route; no pool draw.
std::expected<std::uint64_t, KernelFailure> PlanMmvqPrepared(const LaunchContext& launch,
                                                             const ggml_tensor* node);
std::expected<void, KernelFailure> MulMatQ(LaunchContext& launch, ggml_tensor* node);

// Upstream's fused quantized gate/up product (ggml-cuda.cu:3950-3975,
// ggml_cuda_should_fuse_mul_mat_vec_q at 1794-1820): one
// ggml_cuda_mul_mat_vec_q over the up product's weights and activations,
// with the gate's weights and the GLU fused, writing the GLU. Only where
// upstream fuses: fusion.h MulMatGluFusionAt's structure, MMVQ selected for
// the up product, one output column, F32 activations and output, and a
// device newer than Pascal. The pool scratch is the up product's Q8_1
// activations, prepared once for both products.
bool MulMatVecQGluFusible(const LaunchContext& launch, const ggml_tensor* gate,
                          const ggml_tensor* up, const ggml_tensor* glu);
std::expected<std::uint64_t, KernelFailure> PlanMulMatVecQGlu(const LaunchContext& launch,
                                                              const ggml_tensor* gate,
                                                              const ggml_tensor* up,
                                                              const ggml_tensor* glu);
std::expected<void, KernelFailure> MulMatVecQGlu(LaunchContext& launch, ggml_tensor* gate,
                                                 ggml_tensor* up, ggml_tensor* glu);

// The same raw MMQ inner product and preparation, with device-built
// expert-major tiles for sufficiently large non-FP4 expert products.
// Other supported shapes retain the ordinary MMQ launch.
std::expected<std::uint64_t, KernelFailure> PlanMulMatIdQCompact(const LaunchContext& launch,
                                                                 const ggml_tensor* node);
std::expected<void, KernelFailure> MulMatIdQCompact(LaunchContext& launch, ggml_tensor* node);

// Experimental raw Q2_K product under the same maps and D2S6 Q8 activation
// preparation. ds4's direct-to-register arithmetic differs from GGML's
// half-rounded coefficient arithmetic. No permanent weight replica.
// Automatic selection is restricted to the measured GB10 prefill weights
// (DeepSeek V4's Q2_K down experts) over prefill chunks of
// kDsv4StageMinRows to kDsv4StageMaxRows tokens (validate_ext.h), a
// prompt's last, partial chunk included; direct controls may call the
// bounded generic operation.
bool MulMatIdQ2D2rFits(const LaunchContext& launch, const ggml_tensor* node);
std::expected<std::uint64_t, KernelFailure> PlanMulMatIdQ2D2r(const LaunchContext& launch,
                                                              const ggml_tensor* node);
std::expected<void, KernelFailure> MulMatIdQ2D2r(LaunchContext& launch, ggml_tensor* node);

// The same ordinary MMQ products, in order, with one shared preparation of
// the routing maps and type-specific Q8 activations (ds4's paired MoE
// technique). Both products retain their own weight/output strides and
// sequential fixup workspace. No arithmetic or downstream GLU is fused.
std::expected<std::uint64_t, KernelFailure> PlanMulMatIdQPair(const LaunchContext& launch,
                                                              const ggml_tensor* first,
                                                              const ggml_tensor* second,
                                                              bool compact_experts = false);
std::expected<void, KernelFailure> MulMatIdQPair(LaunchContext& launch, ggml_tensor* first,
                                                 ggml_tensor* second, bool compact_experts = false);
// Experimental (docs/experiments/ds4-prefill-stages): the GB10 IQ2 compact
// pair (validate_ext.h IsMulMatIdQPairGluPair: IQ2_XXS's occupancy-two J64
// launch, IQ2_XS's ordinary J128 one) with swiglu_clamp(gate, up) written
// by the up product (gate first), the up output itself unwritten. Scratch
// as PlanMulMatIdQPair's compact pair.
std::expected<void, KernelFailure> MulMatIdQPairGlu(LaunchContext& launch, ggml_tensor* up,
                                                    ggml_tensor* gate, ggml_tensor* glu);
// Whether this device runs that pair over these operands (GB10 alone has
// these kernels): the planner's device guard (graph_plan.h
// DeviceChoices::pair_glu_fits), so that other devices keep the plain pair.
bool MulMatIdQPairGluSupported(const LaunchContext& launch, const ggml_tensor* up,
                               const ggml_tensor* gate);
// And its quantizing form: the activation's bytes then hold the down
// product's D2S6 input, which MulMatIdQCompactPrequant reads unquantized.
std::expected<void, KernelFailure> MulMatIdQPairGluQ8(LaunchContext& launch, ggml_tensor* up,
                                                      ggml_tensor* gate, ggml_tensor* glu);
std::expected<void, KernelFailure> MulMatIdQCompactPrequant(LaunchContext& launch,
                                                            ggml_tensor* down);
// Experimental: two dense products of one quantized type and one
// activation, sharing its Q8_1 quantization (validate_ext.h
// MulMatQPairDenseFits).
std::expected<std::uint64_t, KernelFailure> PlanMulMatQPairDense(const LaunchContext& launch,
                                                                 const ggml_tensor* a,
                                                                 const ggml_tensor* b);
std::expected<void, KernelFailure> MulMatQPairDense(LaunchContext& launch, ggml_tensor* a,
                                                    ggml_tensor* b);

// Row-invariant products for a speculative verify (D-092; mmvq_rows.cu):
// every output column of a quantized product (up to kRowsMaxColumns
// activation columns, or tokens of a mul_mat_id) is computed with the
// arithmetic of GGML's one-column MMVQ launch, so a verify's row equals the
// decode step it stands for bit for bit; a dense block still reads its
// weight rows once for every column. Weights of the types jitLLM's models
// bring (Q8_0, Q2_K, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_XS, IQ3_XXS, MXFP4,
// Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL); partial row blocks guard every weight
// read and output. The scratch is
// MMVQ's Q8_1 activations.
inline constexpr std::int64_t kRowsMaxColumns = 8;
// The original pinned MMVQ launch's output row block, for checked nodes.
// Used only to prove its unguarded rounded-row reads before submission.
int MmvqRowsPerBlock(const LaunchContext& launch, const ggml_tensor* node);
std::expected<std::uint64_t, KernelFailure> PlanMulMatVecQRows(const LaunchContext& launch,
                                                               const ggml_tensor* node);
std::expected<void, KernelFailure> MulMatVecQRows(LaunchContext& launch, ggml_tensor* node);
// GGML's float vector kernel (MMVF) for up to 8 columns whatever upstream
// would route there (MMF or cuBLAS on the GB10 past one BF16 column or
// three F32 ones): its columns' sums are independent of their count.
std::expected<void, KernelFailure> MulMatVecFRows(LaunchContext& launch, ggml_tensor* node);

// A ggml_mul_mat node with GGML_HINT_SRC0_IS_HADAMARD, as upstream runs it:
// the fast Walsh-Hadamard transform of the activations (ggml_cuda_op_fwht).
std::expected<void, KernelFailure> MulMatHadamard(LaunchContext& launch, ggml_tensor* node);

// Elementwise and row operations, F32 (validate_ext.h). None draws scratch.
std::expected<void, KernelFailure> Unary(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> Scale(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> Clamp(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> Fill(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> Repeat(LaunchContext& launch, ggml_tensor* node);
// ggml_sub and ggml_div nodes with broadcasting, as validate.h's CheckBinary
// takes add and mul.
std::expected<void, KernelFailure> Sub(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> Div(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> Concat(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> SumRows(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> SwiGluClamp(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RopeExt(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> GetRowsExt(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> SetRowsExt(LaunchContext& launch, ggml_tensor* node);

// MoE routing and the indexer's selection, in jitLLM's build without CUB
// (third_party/patches/ggml/0001). Argsort takes the bitonic kernel only,
// refused if a padded row does not fit the device's shared memory. Top-k
// takes upstream's radix select for rows over 1,024 (its HIP path: the k
// indices in no particular order, ties broken by atomics), else the
// bitonic argsort (the k largest in descending order), drawing the scratch
// PlanTopK computes.
std::expected<void, KernelFailure> Argsort(LaunchContext& launch, ggml_tensor* node);
std::expected<std::uint64_t, KernelFailure> PlanTopK(const LaunchContext& launch,
                                                     const ggml_tensor* node);
std::expected<void, KernelFailure> TopK(LaunchContext& launch, ggml_tensor* node);

// Qwen3.8's linear attention: the causal convolution (unfused) and the
// fused gated delta rule.
std::expected<void, KernelFailure> SsmConv(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> GatedDeltaNet(LaunchContext& launch, ggml_tensor* node);

// DeepSeek V4's sparse-attention indexer and hyper-connections.
std::expected<void, KernelFailure> LightningIndexer(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> HcComb(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> HcPre(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> HcPost(LaunchContext& launch, ggml_tensor* node);

// Flash attention through the tensor-core kernels (validate_ext.h
// CheckFlashAttnMma): what the launch does on the context's device. The
// kernel instance is ggml_cuda_flash_attn_ext_mma_f16_case<D, D, columns,
// 8>, columns 1, 2, 4 or 8 as upstream picks them for the query rows
// (fattn.cu:131-164); sparse is the gather of at most n_kv_max unmasked
// cells per row (D 512 and one column only); stream-k splits the cells over
// `blocks` blocks, and the fixup buffer and the mask pre-pass's or the
// sparse indices' buffer come from the pool.
// A node jitllm_ops.h's SetFlashAttnSparseAny marks takes the sparse gather
// whenever its n_kv_max cells are at most half of K's, not only past
// upstream's 4,096.
// An explicit wide_sparse choice permits D256 sparse and shares the
// union of up to eight queries at D256/512. It keeps every original
// per-query mask. Disjoint lists can regress; unknown shapes default to
// the original path. The graph planner records the choice as a distinct
// implementation, so its scratch plan and launch agree.
struct FlashAttnMmaPlan {
  int head = 0;     // D
  int columns = 0;  // ncols1
  int group = 8;    // ncols2
  bool sparse = false;
  bool mask_prepass = false;
  int blocks = 0;
  std::uint64_t scratch = 0;
};
std::expected<FlashAttnMmaPlan, KernelFailure> PlanFlashAttnMma(const LaunchContext& launch,
                                                                const ggml_tensor* node,
                                                                bool wide_sparse = false);
std::expected<void, KernelFailure> FlashAttnMma(LaunchContext& launch, ggml_tensor* node,
                                                bool wide_sparse = false);

// Literal ds4 four-token/G8 HCA core, with native F16 ring bit-copy and
// original dense causal records in one planned scratch scope. Selection
// is default-off and restricted to GB10 HCA graphs of prefill chunks of
// kDsv4HcaMinRows to kDsv4HcaMaxRows rows (tails included) whose ring holds
// at least the rows + 256 raw cells: 256 compressed cells, or 1,024 for
// chunks of up to 2,048 rows.
inline constexpr std::int64_t kDsv4HcaMinRows = kDsv4StageMinRows;
inline constexpr std::int64_t kDsv4HcaMaxRows = kDsv4StageMaxRows;
// Plan sets the core's 88,576-byte shared-memory opt-in before capture;
// Run never changes CUDA function attributes or allocates hidden storage.
bool Dsv4HcaTokentileFits(const LaunchContext& launch, const ggml_tensor* node);
std::expected<std::uint64_t, KernelFailure> PlanDsv4HcaTokentile(const LaunchContext& launch,
                                                                 const ggml_tensor* node);
std::expected<void, KernelFailure> Dsv4HcaTokentile(LaunchContext& launch, ggml_tensor* node);

// Exact GQA2 D256, dense mask, original query tiles4/8/16/32. The mask
// pre-pass needs rounded tile rows when rows>=1024 or multiple sequences.
std::expected<FlashAttnMmaPlan, KernelFailure> PlanFlashAttnMmaGqa2(const LaunchContext& launch,
                                                                    const ggml_tensor* node);
std::expected<void, KernelFailure> FlashAttnMmaGqa2(LaunchContext& launch, ggml_tensor* node);

// The same kernels at head dimension 128 without grouping and without a
// mask (validate_ext.h CheckFlashAttnMma128): the instance
// ggml_cuda_flash_attn_ext_mma_f16_case<128, 128, columns, 1>, columns 8,
// 16, 32 or 64 as upstream's switch_ncols1 picks them for one query head
// per tile, stream-k over the cells and its fixup buffer from the pool.
std::expected<FlashAttnMmaPlan, KernelFailure> PlanFlashAttnMma128(const LaunchContext& launch,
                                                                   const ggml_tensor* node);
std::expected<void, KernelFailure> FlashAttnMma128(LaunchContext& launch, ggml_tensor* node);

}  // namespace jitllm::kernels::ggml

#endif  // JITLLM_KERNELS_GGML_OPS_EXT_H_
