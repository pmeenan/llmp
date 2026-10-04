// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The GGML module's entries in the implementation registry (D-053;
// execution/registry.h), CUDA builds only: a build without the device code
// declares none, so a plan naming one is unsupported there (BP-S4).
//
// For backend-proof P1 the module declares the one operation it offers two
// implementations of, RMSNorm scaled by a weight:
//   ggml.rms_norm_mul.fused    GGML's fused launcher (ops.h RmsNormMul), as
//                              the FP16-F profile runs it;
//   ggml.rms_norm_mul.unfused  rms_norm's launcher, then mul's (ops.h
//                              RmsNormThenMul), as FP16-U runs it;
// and the other operations the FP16 bridge's recorded plan launches, each
// over its nodes in this order (Kernel):
//   ggml.rms_norm                {rms_norm}              ops.h RmsNorm
//   ggml.add                     {add}                   ops.h Add
//   ggml.mul                     {mul}                   ops.h Mul
//   ggml.mul_mat.mmvf            {mul_mat}               ops.h MulMatVecF
//   ggml.mul_mat.mmf             {mul_mat}               ops.h MulMatF
//   ggml.mul_mat.cublas          {mul_mat}               ops.h MulMatCublas
//   ggml.get_rows                {get_rows}              ops.h GetRows
//   ggml.set_rows                {set_rows}              ops.h SetRows
//   ggml.rope.neox               {rope}                  ops.h Rope
//   ggml.rope_set_rows.fused     {rope, set_rows}        ops.h RopeSetRows
//   ggml.soft_max                {soft_max}              ops.h SoftMax
//   ggml.cont                    {cont}                  ops.h Cont
//   ggml.swiglu                  {glu}                   ops.h SwiGlu
//   ggml.geglu                   {glu}                   ops.h GeGlu
//   ggml.mul_mat_geglu.mmvf_fused {gate, up, glu}         ops.h MulMatVecGeGlu
// GeGLU uses GELU-tanh; its primitive fallback is two products then GeGlu.
//   ggml.mul_mat_add.mmvf_fused  {mul_mat, add}          ops.h MulMatVecBias
//   ggml.mul_mat_glu.mmvf_fused  {gate, up, glu}         ops.h MulMatVecGlu
// The three fused ones are FP16-F's; FP16-U runs their parts as separate
// operations (ops.h). The native EXL3 plan (exl3-op-plan.json) also runs
// ggml.get_rows over its BF16 embedding table and
//   ggml.convert                 {cpy}                   ops.h Convert
//   ggml.flash_attn_ext.vec      {flash_attn_ext}        ops.h FlashAttnVec
// For DeepSeek V4 Flash (GGUF) and Qwen3.8 Flash (M3; ops_ext.h,
// validate_ext.h):
//   ggml.mul_mat.mmvq            {mul_mat}               ops_ext.h MulMatVecQ
//   ggml.mul_mat.mmq             {mul_mat}               ops_ext.h MulMatQ
//   ggml.mul_mat.fwht            {mul_mat}               ops_ext.h MulMatHadamard
//   ggml.mul_mat_id.mmvq         {mul_mat_id}            ops_ext.h MulMatVecQ
//   ggml.mul_mat_id.mmq          {mul_mat_id}            ops_ext.h MulMatQ
//   ggml.sub, ggml.div           {sub}, {div}            ops_ext.h Sub, Div
//   ggml.scale                   {scale}                 ops_ext.h Scale
//   ggml.unary                   {unary or sqrt}         ops_ext.h Unary
//   ggml.clamp, ggml.fill        {clamp}, {fill}         ops_ext.h Clamp, Fill
//   ggml.repeat, ggml.concat     {repeat}, {concat}      ops_ext.h Repeat, Concat
//   ggml.sum_rows                {sum_rows}              ops_ext.h SumRows
//   ggml.argsort.bitonic         {argsort}               ops_ext.h Argsort
//   ggml.top_k.radix             {top_k}                 ops_ext.h TopK
//   ggml.swiglu_clamp            {glu}                   ops_ext.h SwiGluClamp
//   ggml.rope.ext                {rope or rope_back}     ops_ext.h RopeExt
//   ggml.get_rows.ext            {get_rows}              ops_ext.h GetRowsExt
//   ggml.set_rows.ext            {set_rows}              ops_ext.h SetRowsExt
//   ggml.ssm_conv                {ssm_conv}              ops_ext.h SsmConv
//   ggml.gated_delta_net         {gated_delta_net}       ops_ext.h GatedDeltaNet
//   ggml.lightning_indexer.wmma  {lightning_indexer}     ops_ext.h LightningIndexer
//   ggml.dsv4_hc_comb, _pre, _post                       ops_ext.h HcComb, HcPre, HcPost
//   ggml.flash_attn_ext.mma      {flash_attn_ext}        ops_ext.h FlashAttnMma
// For Gemma local D256 F16 attention with exactly two query heads per KV head:
//   ggml.flash_attn_ext.vec_d256 {flash_attn_ext}        ops.h FlashAttnVec256
//   ggml.flash_attn_ext.mma_gqa2 {flash_attn_ext}        ops_ext.h FlashAttnMmaGqa2
// For Qwen-Image-2.1's denoiser (M3): multi-head attention at D = 128,
// unmasked, any number of cells:
//   ggml.flash_attn_ext.mma_d128 {flash_attn_ext}        ops_ext.h FlashAttnMma128
// and jitLLM's own operations on GGML tensors, for Qwen3.8's MXFP8 and
// NVFP4 tensors (jitllm_ops.h), each a GGML_OP_CUSTOM node:
//   jitllm.mxfp8.mul_mat_vec     {custom}                RunMxfp8MulMatVec
//   jitllm.mxfp8.dequant         {custom}                RunMxfp8Dequant
//   jitllm.nvfp4.get_rows        {custom}                RunNvfp4Rows
//   jitllm.qrows.get_rows        {custom}                RunQRows
// (GGML's NVFP4 experts take ggml.mul_mat_id.mmvq and .mmq above; the last,
// a GGUF checkpoint's n-gram table in a 32-value block type), and
// jitLLM's fusions of Qwen3.8's GGML nodes, the same arithmetic in the same
// order:
//   jitllm.hc.combine            {custom}                RunHcCombine
//   jitllm.hc.norm               {custom}                RunHcNorm
//   jitllm.hc.mix                {custom}                RunHcMix
//   jitllm.moe.glu               {custom}                RunMoeGlu
//   jitllm.moe.combine           {custom}                RunMoeCombine
//   jitllm.bf16                  {custom}                RunBf16
//   jitllm.gemm.bf16             {custom}                RunGemmBf16
// and a second implementation of GGML's gated_delta_net node, chosen where
// it takes the shape (jitllm_ops.h GatedDeltaNetColumnsFits):
//   jitllm.gated_delta_net.columns {gated_delta_net}     RunGatedDeltaNetColumns
// A speculative verify's row-invariant plan (D-092; graph_plan.h
// DeviceChoices::row_invariant) runs every product of up to 8 columns
// through
//   jitllm.mul_mat.mmvq_rows     {mul_mat}               ops_ext.h MulMatVecQRows
//   jitllm.mul_mat_id.mmvq_rows  {mul_mat_id}            ops_ext.h MulMatVecQRows
//   jitllm.mul_mat.mmvf_rows     {mul_mat}               ops_ext.h MulMatVecFRows
// and DeepSeek's DSpark drafter chains its Markov head on
//   jitllm.argmax                {custom}                RunArgmax
// The quantized products' two implementations are GGML's kernel families,
// which the plan names as upstream would route (ops_ext.h SelectMulMatQ);
// ggml.unary and ggml.rope.ext compute the function the node names.
// Which kernel variant each launches follows from its operands, as
// upstream's launcher chooses it, so a variant names the launcher and its
// rule rather than one kernel.
//
// Each identity covers everything that decides what an implementation
// computes and launches:
//   - the prepared GGML tree's digest, which covers upstream's bytes,
//     jitLLM's patches and the build of GGML's files;
//   - jitLLM's own code in this module: a digest of every file in
//     src/kernels/ggml, written at build time (module_digest.cmake), so any
//     edit here changes every identity the module declares;
//   - the SDK, target, device architecture, build type (NDEBUG, and with it
//     GGML's device asserts), libstdc++'s assertions (D-083) and
//     sanitizers;
//   - the name, and a variant naming the launcher sequence.
// Code outside the module, such as the provider that supplies the stream,
// is not covered: it does not choose what is launched.
// The matrix product's three implementations are GGML's kernel families;
// the plan names the one upstream selects on the device (ops.h
// SelectMulMat; graph_plan.h), and each refuses operands upstream would
// route elsewhere.
//
// A bound plan's implementation becomes a kernel here once, when the plan
// is bound; each launch then runs that kernel's host checks and launchers
// and looks nothing up.

#ifndef JITLLM_KERNELS_GGML_IMPLEMENTATIONS_H_
#define JITLLM_KERNELS_GGML_IMPLEMENTATIONS_H_

#include <cstddef>
#include <expected>
#include <span>
#include <string_view>
#include <vector>

#include "execution/registry.h"
#include "ggml.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/tensors.h"

namespace jitllm::kernels::ggml {

// What this module declares to the registry.
std::vector<execution::Implementation> Implementations();

// The digest of the module's own files, generated at build time.
std::string_view ModuleSourcesDigest();

// One GGML implementation of RMSNorm-mul, over a ggml_rms_norm node and
// the ggml_mul that scales it (ops.h).
class RmsNormMulKernel {
 public:
  // Refused unless `implementation` is one this module declares, identity
  // and all: a stale or foreign declaration never selects a kernel.
  static std::expected<RmsNormMulKernel, KernelFailure> Bind(
      const execution::Implementation& implementation);

  // The implementation's operand checks, on the host (validate.h).
  std::expected<void, KernelFailure> Check(const ggml_tensor* norm, const ggml_tensor* mul) const;
  // Checks, then launches on the context's stream.
  std::expected<void, KernelFailure> Run(LaunchContext& launch, ggml_tensor* norm,
                                         ggml_tensor* mul) const;
  std::string_view name() const;

  struct Entry;

 private:
  explicit RmsNormMulKernel(const Entry& entry) : entry_(&entry) {}

  const Entry* entry_;
};

// One of the module's other implementations (above), over its operation's
// nodes in the order listed there.
// Whether a declared implementation launches cuBLAS on the launch context's
// lent handle, whose stream and workspace are the context stream's: such a
// step stays on lane 0 (graph_plan.h AssignLanes), and on a lane the
// context lends it no handle (launch.h cublas).
bool UsesCublas(std::string_view implementation);

class Kernel {
 public:
  // Refused unless `implementation` is one of those this module declares,
  // identity and all.
  static std::expected<Kernel, KernelFailure> Bind(const execution::Implementation& implementation);

  // The implementation's operand checks, on the host (validate.h); refused
  // unless `nodes` has the operation's number of nodes. Row indices read
  // from device memory are the plan's to bound (ops.h).
  std::expected<void, KernelFailure> Check(std::span<const ggml_tensor* const> nodes) const;
  // Checks, then launches on the context's stream.
  std::expected<void, KernelFailure> Run(LaunchContext& launch,
                                         std::span<ggml_tensor* const> nodes) const;
  std::string_view name() const;
  execution::Operation operation() const;
  // How many nodes the operation takes.
  std::size_t arity() const;

  struct Entry;

 private:
  explicit Kernel(const Entry& entry) : entry_(&entry) {}

  const Entry* entry_;
};

}  // namespace jitllm::kernels::ggml

#endif  // JITLLM_KERNELS_GGML_IMPLEMENTATIONS_H_
