// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Optional checked explicit accesses; ordinary recipe consumers ignore these rows.
#ifndef LLMP_GGML_ACCESS_RECIPE
#define LLMP_GGML_ACCESS_RECIPE(NAME, DESCRIBE)
#define LLMP_GGML_LOCAL_ACCESS_MACRO
#endif

// Shared concrete recipe inventory: key, numerical contract, ordered node arity,
// descriptor validator and launcher. CPU metadata discards only callable expressions.
// No include guard: consumers define both recipe macros before including this file.

LLMP_GGML_NORM_RECIPE("ggml.rms_norm_mul.fused",
                      "ggml_cuda_op_rms_norm_fused; upstream launch configuration",
                      (&CheckRmsNormMul), (&RmsNormMul))

LLMP_GGML_NORM_RECIPE("ggml.rms_norm_mul.unfused",
                      "ggml_cuda_op_rms_norm, then ggml_cuda_op_mul; upstream launch configuration",
                      (&CheckRmsNormThenMul), (&RmsNormThenMul))

LLMP_GGML_RECIPE(kGemmaRouteName, llmp::ir::Primitive(llmp::ir::Op::kRoute),
                 "original ggml_cuda_op_topk_moe; compiled experts/dynamic top-k/clamp2^-14; "
                 "both outputs, full sort backing retained",
                 10, ([](ConstNodes n) -> std::expected<void, KernelFailure> {
                   const auto nodes = Borrow<10>(n);
                   return GemmaRoutingFusionAt(nodes, 0) ? std::expected<void, KernelFailure>{}
                                                         : InvalidGemmaChain();
                 }),
                 ([](LaunchContext& l, Nodes n) -> std::expected<void, KernelFailure> {
                   const auto f = GemmaRoutingFusionAt(n, 0);
                   if (!f) return InvalidGemmaChain();
                   return RunGemmaRouting(l, f->operands);
                 }))

LLMP_GGML_RECIPE(kGemmaReduceName, llmp::ir::Primitive(llmp::ir::Op::kExpertReduce),
                 "original ggml_cuda_op_moe_weighted_reduction; "
                 "F32[width,used,rows], (expert*scale)*weight, ascending selected slots",
                 0, ([](ConstNodes n) -> std::expected<void, KernelFailure> {
                   std::vector<ggml_tensor*> nodes;
                   nodes.reserve(n.size());
                   for (const auto* node : n) nodes.push_back(const_cast<ggml_tensor*>(node));
                   const auto f = GemmaReductionFusionAt(nodes, 0);
                   return f && f->nodes.size() == n.size() ? std::expected<void, KernelFailure>{}
                                                           : InvalidGemmaChain();
                 }),
                 ([](LaunchContext& l, Nodes n) -> std::expected<void, KernelFailure> {
                   const auto f = GemmaReductionFusionAt(n, 0);
                   if (!f) return InvalidGemmaChain();
                   return RunGemmaScaledReduction(l, f->operands);
                 }))

LLMP_GGML_RECIPE(kGemmaNormRopeName, llmp::ir::Fused(llmp::ir::FusedKind::kRmsNormMulRope),
                 "original ggml_cuda_op_rms_norm_mul_rope_fused; F32 full even-width NEOX, "
                 "no direct cache store; default-off checked three-node chain",
                 3, ([](ConstNodes n) { return CheckGemmaNormRope(n[0], n[1], n[2]); }),
                 ([](LaunchContext& l, Nodes n) { return RunGemmaNormRope(l, n[0], n[1], n[2]); }))

LLMP_GGML_RECIPE(
    kGemmaNormAddName, llmp::ir::Fused(llmp::ir::FusedKind::kRmsNormMulAdd),
    "original ggml_cuda_op_rms_norm_fused_add; F32 widths within launcher integer bounds, "
    "default-off checked three-node chain",
    3, ([](ConstNodes n) { return CheckGemmaNormAdd(n[0], n[1], n[2]); }),
    ([](LaunchContext& l, Nodes n) { return RunGemmaNormAdd(l, n[0], n[1], n[2]); }))

LLMP_GGML_RECIPE("ggml.norm", llmp::ir::Primitive(llmp::ir::Op::kLayerNorm),
                 "locked ggml_cuda_op_norm: centered F32 rows, upstream reduction/launch", 1,
                 ([](ConstNodes n) { return CheckLayerNorm(n[0]); }),
                 ([](LaunchContext& l, Nodes n) { return LayerNorm(l, n[0]); }))

LLMP_GGML_RECIPE("ggml.l2_norm", llmp::ir::Primitive(llmp::ir::Op::kL2Norm),
                 "locked ggml_cuda_op_l2_norm; F32 clamped L2, finite normal epsilon squared", 1,
                 ([](ConstNodes n) { return CheckL2Norm(n[0]); }),
                 ([](LaunchContext& l, Nodes n) { return L2Norm(l, n[0]); }))

LLMP_GGML_RECIPE("ggml.rms_norm", llmp::ir::Primitive(llmp::ir::Op::kRmsNorm),
                 "ggml_cuda_op_rms_norm: rms_norm_f32<block, false, false>; upstream launch "
                 "configuration",
                 1, ([](ConstNodes n) { return CheckRmsNorm(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RmsNorm(launch, n[0]); }))

LLMP_GGML_RECIPE(kDsv4QHeadName, llmp::ir::Primitive(llmp::ir::Op::kRope),
                 "QHeadKernel: native 256-thread RMS reduction at width512, explicit F32 "
                 "normalization rounding and normal tail64 RoPE multiply/FMA order",
                 1, ([](ConstNodes n) { return CheckDsv4QHead(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunDsv4QHead(launch, n[0]); }))

LLMP_GGML_RECIPE(kDsv4F16CopyName, llmp::ir::Primitive(llmp::ir::Op::kConvert),
                 "F16CopyKernel: RN F32-to-F16 copy shared by F16-weight cuBLAS products "
                 "(experimental)",
                 1, ([](ConstNodes n) { return CheckDsv4F16Copy(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunDsv4F16Copy(launch, n[0]); }))

LLMP_GGML_RECIPE(kDsv4HcNormF16Name, llmp::ir::Primitive(llmp::ir::Op::kRmsNorm),
                 "HcNormF16Kernel: native 1024-thread flat RMS reduction and F32 scale, then RN "
                 "F16 rows (or the F32 rows) for the HC mix product (experimental)",
                 1, ([](ConstNodes n) { return CheckDsv4HcNormF16(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunDsv4HcNormF16(launch, n[0]); }))

LLMP_GGML_RECIPE(kDsv4HcPostExpertsNormF16Name, llmp::ir::Primitive(llmp::ir::Op::kHcPost),
                 "HcPostNormF16Kernel<true>: the ordered six-slot expert reduction and shared add "
                 "formed in the HC post, then F32 streams and F16 (or F32) mix-input rows "
                 "(experimental)",
                 4, ([](ConstNodes n) {
                   return CheckDsv4HcPostExpertsNormF16(n[0], n[1], n[2], n[3]);
                 }),
                 ([](LaunchContext& launch, Nodes n) {
                   return RunDsv4HcPostExpertsNormF16(launch, n[0], n[1], n[2], n[3]);
                 }))

LLMP_GGML_RECIPE(kDsv4HcPostNormF16Name, llmp::ir::Fused(llmp::ir::FusedKind::kHcNormPost),
                 "HcPostNormF16Kernel: native HC post FMAs into F32 streams, then the native "
                 "flat RMS and RN F16 (or F32) mix-input rows (experimental)",
                 2, ([](ConstNodes n) { return CheckDsv4HcPostNormF16(n[0], n[1]); }),
                 ([](LaunchContext& launch, Nodes n) {
                   return RunDsv4HcPostNormF16(launch, n[0], n[1]);
                 }))

LLMP_GGML_RECIPE(kDsv4OutAName, llmp::ir::Primitive(llmp::ir::Op::kMatMul),
                 "paid raw-Q8 packing, inverse tail64 RoPE, staged F16/HMMA grouped output-A "
                 "into canonical F32 output; borrowed ds4 numerical core",
                 1, ([](ConstNodes n) { return CheckDsv4OutA(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunDsv4OutA(launch, n[0]); }))

LLMP_GGML_RECIPE(kDsv4OutAFastPackName, llmp::ir::Primitive(llmp::ir::Op::kMatMul),
                 "output-A with a coalesced raw-Q8 weight repack (identical bytes), then the same "
                 "staged F16/HMMA grouped product (experimental)",
                 1, ([](ConstNodes n) { return CheckDsv4OutA(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunDsv4OutA(launch, n[0], true); }))

LLMP_GGML_RECIPE("ggml.add", llmp::ir::Primitive(llmp::ir::Op::kAdd),
                 "ggml_cuda_op_add: k_bin_bcast<op_add, float, float, float>; upstream launch "
                 "configuration",
                 1, ([](ConstNodes n) { return CheckBinary(n[0], GGML_OP_ADD); }),
                 ([](LaunchContext& launch, Nodes n) { return Add(launch, n[0]); }))

LLMP_GGML_RECIPE("ggml.mul", llmp::ir::Primitive(llmp::ir::Op::kMul),
                 "ggml_cuda_op_mul: k_bin_bcast<op_mul, float, float, float>; upstream launch "
                 "configuration",
                 1, ([](ConstNodes n) { return CheckBinary(n[0], GGML_OP_MUL); }),
                 ([](LaunchContext& launch, Nodes n) { return Mul(launch, n[0]); }))

LLMP_GGML_RECIPE("ggml.mul_mat.mmvf", llmp::ir::Primitive(llmp::ir::Op::kMatMul),
                 "ggml_cuda_mul_mat_vec_f: mul_mat_vec_f<T, type_acc, ncols, block, false, false> "
                 "as upstream selects; upstream launch configuration",
                 1, ([](ConstNodes n) { return CheckMulMat(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return MulMatVecF(launch, n[0]); }))

LLMP_GGML_RECIPE("ggml.mul_mat.mmf", llmp::ir::Primitive(llmp::ir::Op::kMatMul),
                 "ggml_cuda_mul_mat_f: mul_mat_f<T, warp, cols, nwarps, false> as upstream "
                 "selects; upstream launch configuration",
                 1, ([](ConstNodes n) { return CheckMulMatF(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return MulMatF(launch, n[0]); }))

LLMP_GGML_RECIPE("ggml.mul_mat.cublas", llmp::ir::Primitive(llmp::ir::Op::kMatMul),
                 "GGML's cuBLAS path (mul_mat_cublas.cu): conversions, GemmEx, strided or "
                 "pointer-array batched GEMM on the lent handle, as upstream plans them",
                 1, ([](ConstNodes n) { return CheckMulMatCublasOperands(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return MulMatCublas(launch, n[0]); }))

LLMP_GGML_RECIPE(
    "ggml.get_rows", llmp::ir::Primitive(llmp::ir::Op::kGetRows),
    "ggml_cuda_op_get_rows: F32 rows through k_get_rows_float_vec on 16-byte "
    "vectors, aligned rows and at least 128 blocks, else k_get_rows_float; F16 and BF16 rows "
    "through k_get_rows_float<half or nv_bfloat16, float>; I32 rows keep their type; "
    "upstream launch configuration",
    1, ([](ConstNodes n) { return CheckGetRows(n[0]); }),
    ([](LaunchContext& launch, Nodes n) { return GetRows(launch, n[0]); }))

LLMP_GGML_RECIPE("ggml.set_rows", llmp::ir::Primitive(llmp::ir::Op::kCacheStore),
                 "ggml_cuda_op_set_rows: k_set_rows<float, int64_t, half>; upstream launch "
                 "configuration",
                 1, ([](ConstNodes n) { return CheckSetRows(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return SetRows(launch, n[0]); }))

LLMP_GGML_RECIPE(kSetRowsGroupedName, llmp::ir::Primitive(llmp::ir::Op::kCacheStore),
                 "two through sixteen independent F32/I64/F16 stores; per-store strides and "
                 "fastdiv; primitive ggml_cuda_cast<half>; stack launch arguments; scratch0",
                 0, (&CheckSetRowsGroup), (&RunSetRowsGroup))

LLMP_GGML_RECIPE("ggml.rope.neox", llmp::ir::Primitive(llmp::ir::Op::kRope),
                 "ggml_cuda_op_rope: rope_neox<true, false, float, float>; upstream launch "
                 "configuration",
                 1, ([](ConstNodes n) { return CheckRope(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return Rope(launch, n[0]); }))

LLMP_GGML_RECIPE("ggml.rope_set_rows.fused", llmp::ir::Fused(llmp::ir::FusedKind::kRopeStore),
                 "ggml_cuda_op_rope_fused: rope_neox<true, false, float, half> writing the KV "
                 "destination; upstream launch configuration",
                 2, ([](ConstNodes n) { return CheckRopeSetRows(n[0], n[1]); }),
                 ([](LaunchContext& launch, Nodes n) { return RopeSetRows(launch, n[0], n[1]); }))

LLMP_GGML_RECIPE("ggml.soft_max", llmp::ir::Primitive(llmp::ir::Op::kSoftMax),
                 "ggml_cuda_op_soft_max: soft_max_f32<true, ncols, block, float> for 32 to 4,096 "
                 "columns in powers of two, else soft_max_f32<true, 0, 0, float>, rows in shared "
                 "memory only; upstream launch configuration",
                 1, ([](ConstNodes n) { return CheckSoftMax(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return SoftMax(launch, n[0]); }))

LLMP_GGML_RECIPE(
    "ggml.cont", llmp::ir::Primitive(llmp::ir::Op::kCont),
    "same-type F32/F16 copies: memcpy, pitched copy, scalar or guarded tiled transpose; "
    "I32 memcpy if contiguous, otherwise owned 64-thread scalar copy",
    1, ([](ConstNodes n) -> std::expected<void, KernelFailure> {
      if (auto checked = CheckCont(n[0]); !checked) {
        return std::unexpected(checked.error());
      }
      return {};
    }),
    ([](LaunchContext& launch, Nodes n) { return Cont(launch, n[0]); }))

LLMP_GGML_RECIPE("ggml.swiglu", llmp::ir::Primitive(llmp::ir::Op::kGlu),
                 "ggml_cuda_op_swiglu: unary_gated_op_kernel<op_silu, float>; upstream launch "
                 "configuration",
                 1, ([](ConstNodes n) { return CheckSwiGlu(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return SwiGlu(launch, n[0]); }))

LLMP_GGML_RECIPE("ggml.geglu", llmp::ir::Primitive(llmp::ir::Op::kGlu),
                 "ggml_cuda_op_geglu: unary_gated_op_kernel<op_gelu, float>; upstream launch "
                 "configuration",
                 1, ([](ConstNodes n) { return CheckGeGlu(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return GeGlu(launch, n[0]); }))

LLMP_GGML_RECIPE("ggml.convert", llmp::ir::Primitive(llmp::ir::Op::kConvert),
                 "ggml_cuda_cpy scalar contiguous or strided source to packed destination: "
                 "F16/BF16 to F32 or F32 to F16; upstream launch configuration",
                 1, ([](ConstNodes n) { return CheckConvert(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return Convert(launch, n[0]); }))

LLMP_GGML_RECIPE("ggml.flash_attn_ext.vec", llmp::ir::Primitive(llmp::ir::Op::kAttention),
                 "ggml_cuda_flash_attn_ext_vec_case<64, F16, F16>, forced: "
                 "flash_attn_mask_to_KV_max<ncols> from 1,024 query rows, flash_attn_ext_vec<64, "
                 "1 or 2, F16, F16, false>, flash_attn_combine_results<64> over parallel blocks; "
                 "launch_fattn's launch configuration",
                 1, ([](ConstNodes n) { return CheckFlashAttnVec(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return FlashAttnVec(launch, n[0]); }))

LLMP_GGML_RECIPE("ggml.mul_mat_add.mmvf_fused", llmp::ir::Fused(llmp::ir::FusedKind::kMatMulAdd),
                 "ggml_cuda_mul_mat_vec_f with x_bias, writing the add: mul_mat_vec_f<T, "
                 "type_acc, 1, block, true, false>, the add's precision; upstream launch "
                 "configuration",
                 2, ([](ConstNodes n) { return CheckMulMatVecBias(n[0], n[1]); }),
                 ([](LaunchContext& launch, Nodes n) { return MulMatVecBias(launch, n[0], n[1]); }))

LLMP_GGML_RECIPE("ggml.mul_mat_glu.mmvf_fused", llmp::ir::Fused(llmp::ir::FusedKind::kMatMulGlu),
                 "ggml_cuda_mul_mat_vec_f with gate and SwiGLU, writing the GLU: "
                 "mul_mat_vec_f<T, type_acc, 1, block, true, false>, the GLU's parameters as "
                 "precision; upstream launch configuration",
                 3, ([](ConstNodes n) { return CheckMulMatVecGlu(n[0], n[1], n[2]); }),
                 ([](LaunchContext& launch, Nodes n) {
                   return MulMatVecGlu(launch, n[0], n[1], n[2]);
                 }))

LLMP_GGML_RECIPE("ggml.mul_mat_geglu.mmvf_fused", llmp::ir::Fused(llmp::ir::FusedKind::kMatMulGlu),
                 "ggml_cuda_mul_mat_vec_f with gate and GELU-tanh GLU, writing the GLU: "
                 "mul_mat_vec_f<T, type_acc, 1, block, true, false>, the GLU's parameters as "
                 "precision; upstream launch configuration",
                 3, ([](ConstNodes n) { return CheckMulMatVecGeGlu(n[0], n[1], n[2]); }),
                 ([](LaunchContext& launch, Nodes n) {
                   return MulMatVecGeGlu(launch, n[0], n[1], n[2]);
                 }))

LLMP_GGML_RECIPE("ggml.mul_mat_glu.mmvq_fused", llmp::ir::Fused(llmp::ir::FusedKind::kMatMulGlu),
                 "ggml_cuda_mul_mat_vec_q with gate and GLU, writing the GLU: one Q8_1 "
                 "preparation, mul_mat_vec_q<type, 1, has_fusion>; upstream launch configuration",
                 3, ([](ConstNodes n) {
                   if (auto gate = CheckMulMatQ(n[0]); !gate) return gate;
                   return CheckMulMatQ(n[1]);
                 }),
                 ([](LaunchContext& launch, Nodes n) {
                   return MulMatVecQGlu(launch, n[0], n[1], n[2]);
                 }))

LLMP_GGML_RECIPE("ggml.mul_mat_geglu.mmvq_fused", llmp::ir::Fused(llmp::ir::FusedKind::kMatMulGlu),
                 "ggml_cuda_mul_mat_vec_q with gate and GELU-tanh GLU, writing the GLU: one Q8_1 "
                 "preparation, mul_mat_vec_q<type, 1, has_fusion>; upstream launch configuration",
                 3, ([](ConstNodes n) {
                   if (auto gate = CheckMulMatQ(n[0]); !gate) return gate;
                   return CheckMulMatQ(n[1]);
                 }),
                 ([](LaunchContext& launch, Nodes n) {
                   return MulMatVecQGlu(launch, n[0], n[1], n[2]);
                 }))

LLMP_GGML_RECIPE("ggml.mul_mat.mmvq", llmp::ir::Primitive(llmp::ir::Op::kMatMul),
                 "ggml_cuda_mul_mat_vec_q: quantize_row_q8_1_cuda, then mul_mat_vec_q<type, "
                 "ncols_dst> as upstream selects; upstream launch configuration",
                 1, ([](ConstNodes n) { return CheckMulMatQ(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return MulMatVecQ(launch, n[0]); }))

LLMP_GGML_RECIPE("ggml.mul_mat.mmq", llmp::ir::Primitive(llmp::ir::Op::kMatMul),
                 "ggml_cuda_mul_mat_q: quantize_mmq_q8_1_cuda (or the native FP4 quantization for "
                 "MXFP4 on Blackwell), then mul_mat_q<type, J, fallback> with J and stream-k as "
                 "upstream selects, and its fixup; upstream launch configuration",
                 1, ([](ConstNodes n) { return CheckMulMatQ(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return MulMatQ(launch, n[0]); }))

LLMP_GGML_RECIPE("ggml.mul_mat.fwht", llmp::ir::Primitive(llmp::ir::Op::kMatMul),
                 "ggml_cuda_op_fwht for GGML_HINT_SRC0_IS_HADAMARD: fwht_cuda<n>, the weights "
                 "never read; upstream launch configuration",
                 1, ([](ConstNodes n) { return CheckMulMatHadamard(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return MulMatHadamard(launch, n[0]); }))

LLMP_GGML_RECIPE(
    "ggml.mul_mat_id.mmvq", llmp::ir::Primitive(llmp::ir::Op::kMatMulId),
    "ggml_cuda_mul_mat_vec_q with ids: quantize_row_q8_1_cuda, then mul_mat_vec_q<type, "
    "tokens> over each token's selected experts; upstream launch configuration",
    1, ([](ConstNodes n) { return CheckMulMatIdQ(n[0]); }),
    ([](LaunchContext& launch, Nodes n) { return MulMatVecQ(launch, n[0]); }))

LLMP_GGML_RECIPE("ggml.mul_mat_id.mmq", llmp::ir::Primitive(llmp::ir::Op::kMatMulId),
                 "ggml_cuda_mul_mat_q with ids: ggml_cuda_launch_mm_ids_helper, the activations "
                 "quantized (or scattered) per selected expert, then mul_mat_q<type, J, fallback> "
                 "over each expert's tokens and its fixup; upstream launch configuration",
                 1, ([](ConstNodes n) { return CheckMulMatIdQ(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return MulMatQ(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.mul_mat_id.mmq_pair", llmp::ir::Primitive(llmp::ir::Op::kMatMulId),
                 "two ordinary MMQ expert products sharing one routing map and type-specific "
                 "Q8 preparation, preserving each weight/output stride and sequential fixup",
                 2, ([](ConstNodes n) { return CheckMulMatIdQPair(n[0], n[1]); }),
                 ([](LaunchContext& launch, Nodes n) { return MulMatIdQPair(launch, n[0], n[1]); }))

LLMP_GGML_RECIPE("llmp.mul_mat_id.mmq_compact", llmp::ir::Primitive(llmp::ir::Op::kMatMulId),
                 "ordinary non-FP4 MMQ preparation and inner product with a device-built "
                 "expert-major tile list; original weight/output strides and Q8 arithmetic",
                 1, ([](ConstNodes n) { return CheckMulMatIdQCompact(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return MulMatIdQCompact(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.mul_mat_id.q2_d2r", llmp::ir::Primitive(llmp::ir::Op::kMatMulId),
                 "ds4 down_q2k_d2r_kernel<64,64,raw>: 128x64 tiles; native maps/Q8 D2S6; raw Q2_K",
                 1, ([](ConstNodes n) { return CheckMulMatIdQ2D2r(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return MulMatIdQ2D2r(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.mul_mat_id.mmq_pair_compact", llmp::ir::Primitive(llmp::ir::Op::kMatMulId),
                 "two ordinary non-FP4 MMQ inner products sharing routing and Q8 preparation, "
                 "each launched with a device-built expert-major tile list",
                 2, ([](ConstNodes n) { return CheckMulMatIdQPair(n[0], n[1]); }),
                 ([](LaunchContext& launch, Nodes n) {
                   return MulMatIdQPair(launch, n[0], n[1], true);
                 }))

LLMP_GGML_RECIPE("llmp.mul_mat.mmq_pair_dense", llmp::ir::Primitive(llmp::ir::Op::kMatMul),
                 "two dense MMQ products of one block-quantized non-FP4 type sharing one Q8_1 "
                 "quantization of their activation (experimental)",
                 2, ([](ConstNodes n) { return CheckMulMatQPairDense(n[0], n[1]); }),
                 ([](LaunchContext& launch, Nodes n) {
                   return MulMatQPairDense(launch, n[0], n[1]);
                 }))

LLMP_GGML_RECIPE("llmp.mul_mat_id.mmq_pair_glu", llmp::ir::Primitive(llmp::ir::Op::kMatMulId),
                 "GB10 IQ2 occupancy-two compact pair, gate first, up write-back storing "
                 "swiglu_clamp(gate, up) (experimental)",
                 3, ([](ConstNodes n) { return CheckMulMatIdQPairGlu(n[0], n[1], n[2]); }),
                 ([](LaunchContext& launch, Nodes n) {
                   return MulMatIdQPairGlu(launch, n[0], n[1], n[2]);
                 }))

LLMP_GGML_RECIPE("llmp.mul_mat_id.mmq_pair_glu_q8", llmp::ir::Primitive(llmp::ir::Op::kMatMulId),
                 "the activation write-back pair, its activation stored as the down product's "
                 "D2S6 Q8_1 MMQ input at sorted columns (experimental)",
                 3, ([](ConstNodes n) { return CheckMulMatIdQPairGlu(n[0], n[1], n[2]); }),
                 ([](LaunchContext& launch, Nodes n) {
                   return MulMatIdQPairGluQ8(launch, n[0], n[1], n[2]);
                 }))

LLMP_GGML_RECIPE("llmp.mul_mat_id.mmq_compact_prequant",
                 llmp::ir::Primitive(llmp::ir::Op::kMatMulId),
                 "compact Q2_K MMQ over the pair write-back's D2S6 input, unquantized here "
                 "(experimental)",
                 1, ([](ConstNodes n) { return CheckMulMatIdQCompactPrequant(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) {
                   return MulMatIdQCompactPrequant(launch, n[0]);
                 }))

LLMP_GGML_RECIPE("ggml.sub", llmp::ir::Primitive(llmp::ir::Op::kSub),
                 "ggml_cuda_op_sub: k_bin_bcast<op_sub, float, float, float>; upstream launch "
                 "configuration",
                 1, ([](ConstNodes n) { return CheckBinary(n[0], GGML_OP_SUB); }),
                 ([](LaunchContext& launch, Nodes n) { return Sub(launch, n[0]); }))

LLMP_GGML_RECIPE("ggml.div", llmp::ir::Primitive(llmp::ir::Op::kDiv),
                 "ggml_cuda_op_div: k_bin_bcast<op_div, float, float, float>; upstream launch "
                 "configuration",
                 1, ([](ConstNodes n) { return CheckBinary(n[0], GGML_OP_DIV); }),
                 ([](LaunchContext& launch, Nodes n) { return Div(launch, n[0]); }))

LLMP_GGML_RECIPE(kConv2dTiledF32, llmp::ir::Primitive(llmp::ir::Op::kConv2d),
                 "workspace-derived stripes: pinned F32 im2col, cuBLAS, packed transpose and "
                 "pitched scatter; 64 MiB scratch target; experimental exact-shape opt-in",
                 1, ([](ConstNodes n) -> std::expected<void, KernelFailure> {
                   auto checked = CheckConv2dTiledF32(n[0]);
                   if (!checked) return std::unexpected(checked.error());
                   return {};
                 }),
                 ([](LaunchContext& launch, Nodes n) { return Conv2dTiledF32(launch, n[0]); }))

LLMP_GGML_RECIPE("ggml.conv2d.f32", llmp::ir::Primitive(llmp::ir::Op::kConv2d),
                 "ggml_cuda_op_conv2d direct F32 WHCN; unchanged pinned launch configuration", 1,
                 ([](ConstNodes n) { return CheckConv2dF32(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return Conv2dF32(launch, n[0]); }))

LLMP_GGML_RECIPE(
    kConv2dStridedName, llmp::ir::Primitive(llmp::ir::Op::kConv2d),
    "F32 output, F32 accumulation, stride-aware on-chip F32/BF16 kernel and bias reads", 1,
    ([](ConstNodes n) { return CheckConv2dStrided(n[0]); }),
    ([](LaunchContext& launch, Nodes n) { return Conv2dStrided(launch, n[0]); }))

LLMP_GGML_RECIPE("ggml.upsample.nearest", llmp::ir::Primitive(llmp::ir::Op::kUpsample),
                 "ggml_cuda_op_upscale nearest2 WHCN; unchanged pinned launch configuration", 1,
                 ([](ConstNodes n) { return CheckUpsampleNearest(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return UpsampleNearest(launch, n[0]); }))

LLMP_GGML_RECIPE("ggml.scale", llmp::ir::Primitive(llmp::ir::Op::kScale),
                 "ggml_cuda_op_scale: scale_f32; upstream launch configuration", 1,
                 ([](ConstNodes n) { return CheckScale(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return Scale(launch, n[0]); }))

LLMP_GGML_RECIPE(
    "ggml.unary", llmp::ir::Primitive(llmp::ir::Op::kUnary),
    "ggml_cuda_op_<function> for abs, sgn, neg, silu, GELU-tanh, tanh, relu, sigmoid, exp, "
    "softplus and sqrt: unary_op_kernel<op_<function>, float>; upstream launch "
    "configuration",
    1, ([](ConstNodes n) { return CheckUnary(n[0]); }),
    ([](LaunchContext& launch, Nodes n) { return Unary(launch, n[0]); }))

LLMP_GGML_RECIPE("ggml.clamp", llmp::ir::Primitive(llmp::ir::Op::kClamp),
                 "ggml_cuda_op_clamp: op_clamp_kernel<float>; upstream launch configuration", 1,
                 ([](ConstNodes n) { return CheckClamp(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return Clamp(launch, n[0]); }))

LLMP_GGML_RECIPE("ggml.fill", llmp::ir::Primitive(llmp::ir::Op::kFill),
                 "ggml_cuda_op_fill: fill_kernel<float or half>; upstream launch configuration", 1,
                 ([](ConstNodes n) { return CheckFill(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return Fill(launch, n[0]); }))

LLMP_GGML_RECIPE("ggml.repeat", llmp::ir::Primitive(llmp::ir::Op::kRepeat),
                 "ggml_cuda_op_repeat: k_bin_bcast<op_repeat, float, float, float>; upstream "
                 "launch configuration",
                 1, ([](ConstNodes n) { return CheckRepeat(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return Repeat(launch, n[0]); }))

LLMP_GGML_RECIPE("ggml.concat", llmp::ir::Primitive(llmp::ir::Op::kConcat),
                 "ggml_cuda_op_concat: concat_cont<T, dim> per sample for contiguous operands, two "
                 "copies along the samples, else concat_non_cont<T, dim>; upstream launch "
                 "configuration",
                 1, ([](ConstNodes n) { return CheckConcat(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return Concat(launch, n[0]); }))

LLMP_GGML_RECIPE("ggml.sum_rows", llmp::ir::Primitive(llmp::ir::Op::kSumRows),
                 "ggml_cuda_op_sum_rows: reduce_rows_f32<false>, 512 threads per row below two "
                 "rows per multiprocessor, else 32 or 128; upstream launch configuration",
                 1, ([](ConstNodes n) { return CheckSumRows(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return SumRows(launch, n[0]); }))

LLMP_GGML_RECIPE("ggml.argsort.bitonic", llmp::ir::Primitive(llmp::ir::Op::kArgsort),
                 "ggml_cuda_op_argsort for rows of at most 1,024 that fit shared memory: "
                 "k_argsort_f32_i32<order>; upstream launch configuration",
                 1, ([](ConstNodes n) { return CheckArgsort(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return Argsort(launch, n[0]); }))

LLMP_GGML_RECIPE(
    "ggml.top_k.radix", llmp::ir::Primitive(llmp::ir::Op::kTopK),
    "ggml_cuda_op_top_k without CUB: top_k_radix_cuda (8-bit radix select, unordered) "
    "for rows over 1,024, else k_argsort_f32_i32<DESC> and a pitched copy of the first "
    "k; upstream launch configuration",
    1, ([](ConstNodes n) { return CheckTopK(n[0]); }),
    ([](LaunchContext& launch, Nodes n) { return TopK(launch, n[0]); }))

LLMP_GGML_RECIPE("ggml.swiglu_clamp", llmp::ir::Primitive(llmp::ir::Op::kGlu),
                 "ggml_cuda_op_swiglu_clamp: swiglu_clamp_kernel<float>; upstream launch "
                 "configuration",
                 1, ([](ConstNodes n) { return CheckSwiGluClamp(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return SwiGluClamp(launch, n[0]); }))

LLMP_GGML_RECIPE("ggml.rope.ext", llmp::ir::Primitive(llmp::ir::Op::kRope),
                 "ggml_cuda_op_rope or ggml_cuda_op_rope_back: rope_norm, rope_neox or rope_multi "
                 "<forward, false, float, float> by the node's mode, with its offset and YaRN; "
                 "upstream launch configuration",
                 1, ([](ConstNodes n) { return CheckRopeExt(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RopeExt(launch, n[0]); }))

LLMP_GGML_RECIPE("ggml.get_rows.ext", llmp::ir::Primitive(llmp::ir::Op::kGetRows),
                 "ggml_cuda_op_get_rows: k_get_rows<qk, qr, dequantize> for Q8_0, k_get_rows_kq<"
                 "dequantize_type> for the k- and i-quants and MXFP4, k_get_rows_float<int32_t, "
                 "int32_t> for I32; upstream launch configuration",
                 1, ([](ConstNodes n) { return CheckGetRowsExt(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return GetRowsExt(launch, n[0]); }))

LLMP_GGML_RECIPE("ggml.set_rows.ext", llmp::ir::Primitive(llmp::ir::Op::kCacheStore),
                 "ggml_cuda_op_set_rows: k_set_rows<src, I32 or I64, dst> for F32 into F32 or F16 "
                 "and F16 into F16; upstream launch configuration",
                 1, ([](ConstNodes n) { return CheckSetRowsExt(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return SetRowsExt(launch, n[0]); }))

LLMP_GGML_RECIPE(
    "ggml.ssm_conv", llmp::ir::Primitive(llmp::ir::Op::kSsmConv),
    "ggml_cuda_op_ssm_conv unfused: ssm_conv_f32<false, 128, conv> up to 32 tokens, "
    "else ssm_conv_long_token_f32<false, 128, conv, 32>; upstream launch configuration",
    1, ([](ConstNodes n) { return CheckSsmConv(n[0]); }),
    ([](LaunchContext& launch, Nodes n) { return SsmConv(launch, n[0]); }))

LLMP_GGML_RECIPE("ggml.gated_delta_net", llmp::ir::Primitive(llmp::ir::Op::kGatedDeltaNet),
                 "ggml_cuda_op_gated_delta_net: gated_delta_net_cuda<S, KDA, snapshots>; upstream "
                 "launch configuration",
                 1, ([](ConstNodes n) { return CheckGatedDeltaNet(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return GatedDeltaNet(launch, n[0]); }))

LLMP_GGML_RECIPE(
    "ggml.lightning_indexer.wmma", llmp::ir::Primitive(llmp::ir::Op::kIndexScore),
    "ggml_cuda_lightning_indexer on tensor cores: lightning_indexer_kernel_wmma<8, 32, "
    "128, heads, F16>; upstream launch configuration",
    1, ([](ConstNodes n) { return CheckLightningIndexer(n[0]); }),
    ([](LaunchContext& launch, Nodes n) { return LightningIndexer(launch, n[0]); }))

LLMP_GGML_RECIPE("ggml.dsv4_hc_comb", llmp::ir::Primitive(llmp::ir::Op::kHcMix),
                 "ggml_cuda_op_dsv4_hc_comb: dsv4_hc_comb_f32; upstream launch configuration", 1,
                 ([](ConstNodes n) { return CheckHcComb(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return HcComb(launch, n[0]); }))

LLMP_GGML_RECIPE("ggml.dsv4_hc_pre", llmp::ir::Primitive(llmp::ir::Op::kHcPre),
                 "ggml_cuda_op_dsv4_hc_pre: dsv4_hc_pre_f32; upstream launch configuration", 1,
                 ([](ConstNodes n) { return CheckHcPre(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return HcPre(launch, n[0]); }))

LLMP_GGML_RECIPE("ggml.dsv4_hc_post", llmp::ir::Primitive(llmp::ir::Op::kHcPost),
                 "ggml_cuda_op_dsv4_hc_post: dsv4_hc_post_f32; upstream launch configuration", 1,
                 ([](ConstNodes n) { return CheckHcPost(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return HcPost(launch, n[0]); }))

LLMP_GGML_RECIPE("ggml.flash_attn_ext.vec_d256", llmp::ir::Primitive(llmp::ir::Op::kAttention),
                 "pinned D256/F16 vector attention, exact GQA2; parallel split and original fixup",
                 1, ([](ConstNodes n) { return CheckFlashAttnVec256(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return FlashAttnVec256(launch, n[0]); }))

LLMP_GGML_RECIPE("ggml.flash_attn_ext.mma_gqa2", llmp::ir::Primitive(llmp::ir::Op::kAttention),
                 "pinned D256/group2 MMA, query tiles4/8/16/32; dense mask and stream-k fixup", 1,
                 ([](ConstNodes n) { return CheckFlashAttnMmaGqa2(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return FlashAttnMmaGqa2(launch, n[0]); }))

LLMP_GGML_RECIPE(
    "ggml.flash_attn_ext.mma", llmp::ir::Primitive(llmp::ir::Op::kAttention),
    "ggml_cuda_flash_attn_ext_mma_f16_case<D, D, 1, 2, 4 or 8, 8> for D 256 and 512 as "
    "switch_ncols1 picks, the sparse gather at D 512 as upstream decides, launch_fattn "
    "with stream-k and its fixup; upstream launch configuration",
    1, ([](ConstNodes n) { return CheckFlashAttnMma(n[0]); }),
    ([](LaunchContext& launch, Nodes n) { return FlashAttnMma(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.dsv4.hca_tokentile", llmp::ir::Primitive(llmp::ir::Op::kAttention),
                 "ds4 attention_tokentile_hmma_kernel: four tokens/G8, M32/R32, 16 warps; "
                 "F32 Q rounded to F16, original F32 QK/softmax/PV accumulation and F16 "
                 "probabilities; native F16 ring mirror and original dense causal records",
                 1, ([](ConstNodes n) { return CheckDsv4HcaTokentile(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return Dsv4HcaTokentile(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.flash_attn_ext.mma_wide", llmp::ir::Primitive(llmp::ir::Op::kAttention),
                 "explicit sparse query-union choice at D256/512, one or eight query columns; "
                 "original per-query masks and F16 KV, stream-k with fixup",
                 1, ([](ConstNodes n) { return CheckFlashAttnMma(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return FlashAttnMma(launch, n[0], true); }))

LLMP_GGML_RECIPE(kAttentionF32, llmp::ir::Primitive(llmp::ir::Op::kAttention),
                 "original F32 MHA and additive mask, privately F16-packed/padded; existing "
                 "D128 MMA with F32 accumulation/output, qualified finite F16 operand range",
                 1, ([](ConstNodes n) -> std::expected<void, KernelFailure> {
                   auto checked = CheckAttentionF32(n[0]);
                   if (!checked) return std::unexpected(checked.error());
                   return {};
                 }),
                 ([](LaunchContext& launch, Nodes n) { return AttentionF32(launch, n[0]); }))

LLMP_GGML_RECIPE("ggml.flash_attn_ext.mma_common", llmp::ir::Primitive(llmp::ir::Op::kAttention),
                 "upstream group-one MMA at D64/80/96/112/128/192, F16 KV, mask and arbitrary GQA; "
                 "query tiles8/32 (D128 retains8/16/32/64), original stream-k fixup",
                 1, ([](ConstNodes n) { return CheckFlashAttnMmaCommon(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return FlashAttnMmaCommon(launch, n[0]); }))

LLMP_GGML_RECIPE("ggml.flash_attn_ext.vec_common", llmp::ir::Primitive(llmp::ir::Op::kAttention),
                 "upstream F16 vector attention at D64/128/192, padded KV, masks and arbitrary GQA",
                 1, ([](ConstNodes n) { return CheckFlashAttnVecCommon(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return FlashAttnVecCommon(launch, n[0]); }))

LLMP_GGML_RECIPE("ggml.flash_attn_ext.mma_d128", llmp::ir::Primitive(llmp::ir::Op::kAttention),
                 "ggml_cuda_flash_attn_ext_mma_f16_case<128, 128, 8, 16, 32 or 64, 1> as "
                 "switch_ncols1 picks, no mask, launch_fattn with stream-k and its fixup; upstream "
                 "launch configuration",
                 1, ([](ConstNodes n) { return CheckFlashAttnMma128(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return FlashAttnMma128(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.mxfp8.mul_mat_vec", llmp::ir::Primitive(llmp::ir::Op::kMatMul),
                 "Mxfp8Gemv<columns 1-8, rows 1/4/2 a warp>: each row's sums as one warp a row's, "
                 "16-code vectors, F32 block sums; PDL, the first weights prefetched into L2",
                 1, ([](ConstNodes n) { return CheckMxfp8MulMatVec(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunMxfp8MulMatVec(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.nvfp4.get_rows", llmp::ir::Primitive(llmp::ir::Op::kGetRows),
                 "Nvfp4RowsKernel: a block an id, a thread a value", 1,
                 ([](ConstNodes n) { return CheckNvfp4Rows(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunNvfp4Rows(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.qrows.get_rows", llmp::ir::Primitive(llmp::ir::Op::kGetRows),
                 "QRowsKernel<type>: a block an id, a thread a value, GGML's dequantize.cuh", 1,
                 ([](ConstNodes n) { return CheckQRows(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunQRows(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.hc.combine", llmp::ir::Fused(llmp::ir::FusedKind::kHcCombine),
                 "HcCombineKernel: four columns a thread", 1,
                 ([](ConstNodes n) { return CheckHcCombine(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunHcCombine(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.hc.norm", llmp::ir::Fused(llmp::ir::FusedKind::kHcNorm),
                 "HcNormKernel<F32 or BF16>: a stream of a token a 1,024-thread block, "
                 "rms_norm_f32<1024>'s reduction",
                 1, ([](ConstNodes n) { return CheckHcNorm(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunHcNorm(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.hc.mix", llmp::ir::Primitive(llmp::ir::Op::kHcPre),
                 "HcMixKernel: a token a 1,024-thread block, rms_norm_f32<1024>'s reduction per "
                 "stream",
                 1, ([](ConstNodes n) { return CheckHcMix(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunHcMix(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.moe.glu", llmp::ir::Fused(llmp::ir::FusedKind::kMoeGlu),
                 "MoeGluKernel: four columns a thread", 1,
                 ([](ConstNodes n) { return CheckMoeGlu(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunMoeGlu(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.moe.combine", llmp::ir::Fused(llmp::ir::FusedKind::kMoeCombine),
                 "MoeCombineKernel: four columns a thread, experts in order", 1,
                 ([](ConstNodes n) { return CheckMoeCombine(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunMoeCombine(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.bf16", llmp::ir::Primitive(llmp::ir::Op::kConvert),
                 "Bf16Kernel: __float2bfloat16, an element a thread", 1,
                 ([](ConstNodes n) { return CheckBf16(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunBf16(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.gemm.bf16", llmp::ir::Primitive(llmp::ir::Op::kMatMul),
                 "cublasGemmEx BF16 x BF16 into declared F32 or BF16 storage, CUBLAS_COMPUTE_32F, "
                 "default tensor-op algorithm; "
                 "GemvBf16 nodes of one column GemvBf16Rows or GemvBf16Split (PDL, F32 sums)",
                 1, ([](ConstNodes n) { return CheckGemmBf16(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunGemmBf16(launch, n[0]); }))

LLMP_GGML_RECIPE(kGemmBf16LtName, llmp::ir::Primitive(llmp::ir::Op::kMatMul),
                 "llmp.gemm.bf16 with the GB10's pinned cuBLASLt algorithm where its shape has one "
                 "(mul_mat_cublas.cu kGb10LtPins, the image's products), else cublasGemmEx",
                 1, ([](ConstNodes n) { return CheckGemmBf16(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunGemmBf16(launch, n[0]); }))

LLMP_GGML_RECIPE(
    kAttentionMaskPackName, llmp::ir::Fused(llmp::ir::FusedKind::kAttentionMaskPack),
    "PackAttentionMaskWholeKernel: a packed row a block, F16 of the mask, -inf past it, "
    "a whole row marked by its first element",
    1, ([](ConstNodes n) { return CheckAttentionMaskPack(n[0]); }),
    ([](LaunchContext& launch, Nodes n) { return RunAttentionMaskPack(launch, n[0]); }))

LLMP_GGML_RECIPE(kRopeTableName, llmp::ir::Fused(llmp::ir::FusedKind::kRopeTable),
                 "RopeTableKernel: a row a block, each pair's cos and sin of its section's angle",
                 1, ([](ConstNodes n) -> std::expected<void, KernelFailure> {
                   if (auto args = CheckRopeTable(n[0]); !args)
                     return std::unexpected(args.error());
                   return {};
                 }),
                 ([](LaunchContext& launch, Nodes n) { return RunRopeTable(launch, n[0]); }))

LLMP_GGML_RECIPE(
    kQkAttentionName, llmp::ir::Fused(llmp::ir::FusedKind::kQkAttention),
    "FlashForwardBf16RowMajorKernel<2, out, norm q>: BF16 row-major heads, the query's "
    "RMS norm and table rotation in the kernel, a packed F16 mask",
    1, ([](ConstNodes n) { return CheckQkAttention(n[0]); }),
    ([](LaunchContext& launch, Nodes n) { return RunQkAttention(launch, n[0]); }))

LLMP_GGML_RECIPE(kLayerNormModulateSplitName,
                 llmp::ir::Fused(llmp::ir::FusedKind::kLayerNormModulateSplit),
                 "LayerNormModulateRows(4096)Kernel: a row a block, its layer norm times one plus "
                 "its row range's conditioning vector, into F32 or BF16",
                 1, ([](ConstNodes n) { return CheckLayerNormModulateSplit(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) {
                   return RunLayerNormModulateSplit(launch, n[0]);
                 }))

LLMP_GGML_RECIPE(kGatedResidualSplitName, llmp::ir::Fused(llmp::ir::FusedKind::kGatedResidualSplit),
                 "GatedResidualSplitKernel: an element a thread, carry plus branch times tanh of "
                 "its row range's gate",
                 1, ([](ConstNodes n) { return CheckGatedResidualSplit(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) {
                   return RunGatedResidualSplit(launch, n[0]);
                 }))

LLMP_GGML_RECIPE(kGatedResidualModulateSplitName,
                 llmp::ir::Fused(llmp::ir::FusedKind::kGatedResidualModulateSplit),
                 "GatedResidualLayerNormModulateRows(4096)Kernel: a row a block, the residual's "
                 "sum written and kept, its layer norm times one plus its row range's vector",
                 2, ([](ConstNodes n) { return CheckGatedResidualModulateSplit(n[0], n[1]); }),
                 ([](LaunchContext& launch, Nodes n) {
                   return RunGatedResidualModulateSplit(launch, n[0], n[1]);
                 }))

LLMP_GGML_RECIPE(kDupUpAddName, llmp::ir::Fused(llmp::ir::FusedKind::kDupUpAdd),
                 "DupUpAddF32Kernel: an output element a thread, x plus its shortcut channel's "
                 "duplicate at that pixel",
                 1, ([](ConstNodes n) { return CheckDupUpAdd(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunDupUpAdd(launch, n[0]); }))

LLMP_GGML_RECIPE(kConv1x1Name, llmp::ir::Primitive(llmp::ir::Op::kConv2d),
                 "cublasGemmEx F32 over the pixels ([P, Ci] by [Ci, Co], default algorithm), then "
                 "AddBiasWhcF32Kernel",
                 1, ([](ConstNodes n) { return CheckConv1x1(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunConv1x1(launch, n[0]); }))

LLMP_GGML_RECIPE(kSwiGluBf16Name, llmp::ir::Primitive(llmp::ir::Op::kGlu),
                 "SwiGluToBf16Kernel: silu(gate) * up of F32 or BF16 operands into BF16, eight "
                 "values a thread where aligned",
                 1, ([](ConstNodes n) { return CheckSwiGluBf16(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunSwiGluBf16(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.gated_delta_net.columns", llmp::ir::Primitive(llmp::ir::Op::kGatedDeltaNet),
                 "GdnColumnsKernel<4>: gated_delta_net_cuda<128>'s per-column arithmetic, four "
                 "value columns a warp, four warps a block",
                 1, ([](ConstNodes n) { return CheckGatedDeltaNetColumns(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) {
                   return RunGatedDeltaNetColumns(launch, n[0]);
                 }))

LLMP_GGML_RECIPE("llmp.gated_delta_net.lanes", llmp::ir::Primitive(llmp::ir::Op::kGatedDeltaNet),
                 "GdnLanesKernel: a value column over 8 lanes of 16 rows, 64 columns a block, "
                 "16-token chunks staged in shared memory by asynchronous copies",
                 1, ([](ConstNodes n) { return CheckGatedDeltaNetLanes(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) {
                   return RunGatedDeltaNetLanes(launch, n[0]);
                 }))

LLMP_GGML_RECIPE("llmp.gdn.conv", llmp::ir::Primitive(llmp::ir::Op::kSsmConv),
                 "GdnConvKernel<F32 or BF16 rows>: a head of a token a 128-thread block; "
                 "ssm_conv, silu, and rms_norm_f32<256>'s reduction for the query and key heads",
                 1, ([](ConstNodes n) { return CheckGdnConv(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunGdnConv(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.gdn.norm_gate", llmp::ir::Fused(llmp::ir::FusedKind::kNormGate),
                 "GdnNormGateKernel<F32 or BF16 out, F32 or BF16 z>: a head a warp, "
                 "rms_norm_f32<256>'s reduction; GdnNormGateMxfp8Kernel<z> into MXFP8 rows",
                 1, ([](ConstNodes n) { return CheckGdnNormGate(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunGdnNormGate(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.moe.route", llmp::ir::Primitive(llmp::ir::Op::kExpertSort),
                 "RouteCount, RouteScan, RouteAssign: 64-token chunks, slots in token order", 1,
                 ([](ConstNodes n) { return CheckMoeRoute(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunMoeRoute(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.moe.quantize", llmp::ir::Primitive(llmp::ir::Op::kPrepare),
                 "QuantizeRows: quantize_mmq_nvfp4's row and block scales, a token a block", 1,
                 ([](ConstNodes n) { return CheckMoeQuantize(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunMoeQuantize(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.moe.gemm.cutlass", llmp::ir::Primitive(llmp::ir::Op::kMatMulId),
                 "CUTLASS 4.7.1 Sm120 block-scaled NVFP4 grouped GEMM, KernelPtrArrayTmaWarp"
                 "SpecializedPingpong, tile 128x128x256, cluster 1x1x1, BF16 LinearCombination "
                 "epilogue; device-side problem shapes from SetupGroups",
                 1, ([](ConstNodes n) { return CheckMoeGemm(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunMoeGemm(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.moe.glu_quantize", llmp::ir::Fused(llmp::ir::FusedKind::kMoeGlu),
                 "GluQuantizeRows: SwiGLU then quantize_mmq_nvfp4's scales, a row a block", 1,
                 ([](ConstNodes n) { return CheckMoeGluQuantize(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunMoeGluQuantize(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.moe.combine_sorted", llmp::ir::Fused(llmp::ir::FusedKind::kMoeCombine),
                 "CombineSorted: four columns a thread, experts in order", 1,
                 ([](ConstNodes n) { return CheckMoeCombineSorted(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunMoeCombineSorted(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.moe.gemv", llmp::ir::Primitive(llmp::ir::Op::kMatMulId),
                 "Gemv: a warp an output row of a slot, 16-value blocks a lane, F32 activations; "
                 "PDL",
                 1, ([](ConstNodes n) { return CheckMoeGemv(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunMoeGemv(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.mul_mat.mmvq_rows", llmp::ir::Primitive(llmp::ir::Op::kMatMul),
                 "MulMatVecQRowsKernel<type, columns 1-8>: quantize_row_q8_1_cuda, then GGML's "
                 "mul_mat_vec_q body with the one-column launch's warps, rows per block, small-K "
                 "and halved iterations for every column; GGML's device flags",
                 1, ([](ConstNodes n) { return CheckMulMatQ(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return MulMatVecQRows(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.mul_mat_id.mmvq_rows", llmp::ir::Primitive(llmp::ir::Op::kMatMulId),
                 "MulMatVecQRowsKernel<type, 1, per token>: quantize_row_q8_1_cuda, then GGML's "
                 "one-token mul_mat_vec_q launch with ids for every token, over grid z",
                 1, ([](ConstNodes n) { return CheckMulMatIdQ(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return MulMatVecQRows(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.mul_mat.mmvf_rows", llmp::ir::Primitive(llmp::ir::Op::kMatMul),
                 "ggml_cuda_mul_mat_vec_f: mul_mat_vec_f<T, type_acc, columns 1-8, block, false, "
                 "false> whatever upstream would route; upstream launch configuration",
                 1, ([](ConstNodes n) { return CheckMulMatVecFRows(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return MulMatVecFRows(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.argmax", llmp::ir::Primitive(llmp::ir::Op::kTopK),
                 "ArgmaxKernel<probability>: a block a row, the highest value, the lowest index "
                 "among equals; optionally its softmax probability, a second pass in a fixed "
                 "order",
                 1, ([](ConstNodes n) { return CheckArgmax(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunArgmax(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.mxfp8.quantize", llmp::ir::Primitive(llmp::ir::Op::kPrepare),
                 "Mxfp8QuantizeKernel<F32 or BF16>: a 32-value block a thread, E8M0 scale "
                 "2^ceil(log2(amax / 448)), E4M3 codes rounded to nearest, swizzled scales",
                 1, ([](ConstNodes n) { return CheckMxfp8Quantize(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunMxfp8Quantize(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.mxfp8.swizzle", llmp::ir::Primitive(llmp::ir::Op::kConvert),
                 "Mxfp8SwizzleKernel: a scale a thread, rows padded to 128", 1,
                 ([](ConstNodes n) { return CheckMxfp8Swizzle(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunMxfp8Swizzle(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.mxfp8.gemm.cutlass", llmp::ir::Primitive(llmp::ir::Op::kMatMul),
                 "CUTLASS 4.7.1 Sm120 block-scaled MXFP8 GEMM, KernelTmaWarpSpecializedPingpong, "
                 "tile 128x128x128, cluster 1x1x1, F32 or BF16 LinearCombination epilogue, tiles "
                 "swizzled 8 along N past 4,096 rows",
                 1, ([](ConstNodes n) { return CheckMxfp8Gemm(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunMxfp8Gemm(launch, n[0]); }))

LLMP_GGML_RECIPE(
    "llmp.hc.prep", llmp::ir::Fused(llmp::ir::FusedKind::kHcNorm),
    "HcPrepKernel<combine, inject>: a token a block, a float4 column of every stream "
    "a thread; the combine, the streams' RMS norms into BF16, the inject logits. Up to "
    "8 tokens, on compute capability 9.0 and later, HcPrepClusterKernel: a token a "
    "cluster of 8 blocks, the sums through distributed shared memory in block order",
    1, ([](ConstNodes n) { return CheckHcPrep(n[0]); }),
    ([](LaunchContext& launch, Nodes n) { return RunHcPrep(launch, n[0]); }))

LLMP_GGML_RECIPE(
    kSpatialChannelNormName, llmp::ir::Fused(llmp::ir::FusedKind::kSpatialChannelNorm),
    "SpatialChannelNormTiledF32Kernel up to 1024 channels: 32 pixels a block, their "
    "channels dealt over 8 warps (16 above 512) and held in registers (one read), the "
    "sums of squares met in shared memory; above, SpatialChannelNormF32Kernel: a pixel "
    "a thread. Each channel normed, scaled by the factor and gamma, SiLU where asked",
    1, ([](ConstNodes n) { return CheckSpatialChannelNorm(n[0]); }),
    ([](LaunchContext& launch, Nodes n) { return RunSpatialChannelNorm(launch, n[0]); }))

LLMP_GGML_RECIPE(kNormRopeSectionsName,
                 llmp::ir::Fused(llmp::ir::FusedKind::kRmsNormMulRopeSections),
                 "RmsNormMulRope3dKernel: a row a block, a head a warp; each head's RMS norm times "
                 "its F32 or BF16 weight, then each section's pairs rotated by its own positions",
                 1, ([](ConstNodes n) -> std::expected<void, KernelFailure> {
                   if (auto args = CheckNormRopeSections(n[0]); !args)
                     return std::unexpected(args.error());
                   return {};
                 }),
                 ([](LaunchContext& launch, Nodes n) { return RunNormRopeSections(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.hc.lo", llmp::ir::Primitive(llmp::ir::Op::kUnary),
                 "HcLoKernel: silu(lo / hc) into BF16, an element a thread", 1,
                 ([](ConstNodes n) { return CheckHcLo(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunHcLo(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.hc.mix_bf16", llmp::ir::Primitive(llmp::ir::Op::kHcPre),
                 "HcMixBf16Kernel: eight columns of a token a thread, BF16 streams and logits", 1,
                 ([](ConstNodes n) { return CheckHcMixBf16(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunHcMixBf16(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.moe.router", llmp::ir::Primitive(llmp::ir::Op::kArgsort),
                 "MoeRouterKernel: a token a warp; softmax, top experts by warp argmax, their "
                 "weights renormalized, the shared expert's gate logit",
                 1, ([](ConstNodes n) { return CheckMoeRouter(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunMoeRouter(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.gdn.history", llmp::ir::Primitive(llmp::ir::Op::kCont),
                 "GdnHistoryKernel<F32 or BF16>: a channel a thread, the last rows transposed, "
                 "after the old history's last taps where the rows are fewer",
                 1, ([](ConstNodes n) { return CheckGdnHistory(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunGdnHistory(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.gdn.gates", llmp::ir::Fused(llmp::ir::FusedKind::kGdnGates),
                 "GdnGatesKernel: paired sigmoid(beta) and softplus(alpha + dt_bias) * ssm_a, "
                 "the original GGML F32 rounding points",
                 1, ([](ConstNodes n) { return CheckGdnGates(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunGdnGates(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.gdn.step", llmp::ir::Primitive(llmp::ir::Op::kGatedDeltaNet),
                 "GdnColumnsKernel<4> over the state in place: the columns kernel's arithmetic, "
                 "the new state written over the old, the attention output alone",
                 1, ([](ConstNodes n) { return CheckGdnStep(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunGdnStep(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.qsa.prep", llmp::ir::Primitive(llmp::ir::Op::kRope),
                 "QsaPrepKernel: a head of a token a warp; rms_norm times the weight, then "
                 "rope_multi's NEOX pairs at the token's position over 64 dimensions",
                 1, ([](ConstNodes n) { return CheckQsaPrep(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunQsaPrep(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.qsa.gate_quantize", llmp::ir::Primitive(llmp::ir::Op::kPrepare),
                 "QsaGateQuantizeKernel: a 32-value block a thread, attention times "
                 "sigmoid(gate), MXFP8",
                 1, ([](ConstNodes n) { return CheckQsaGateQuantize(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunQsaGateQuantize(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.qsa.pool", llmp::ir::Primitive(llmp::ir::Op::kRope),
                 "QsaPoolKernel: a block the chunk completes a warp; its raw keys summed in cell "
                 "order over the ratio, then qsa.prep's norm and rotation at its first position, "
                 "BF16 into the block keys in place",
                 1, ([](ConstNodes n) { return CheckQsaPool(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunQsaPool(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.qsa.topk", llmp::ir::Primitive(llmp::ir::Op::kTopK),
                 "QsaQueryBf16Kernel, then QsaScoreVecKernel (a block a thread, up to 16 tokens) "
                 "or QsaScoreMmaKernel (BF16 m16n8k16, 128 block keys in registers, 16-token "
                 "tiles), keys in scratch; QsaSelectTileKernel (a token's 8,192-block tile a "
                 "256-thread block, four-pass radix select over per-warp histograms, ties "
                 "to the lower cell) and past "
                 "one tile QsaSelectMergeKernel over the tiles' candidates",
                 1, ([](ConstNodes n) { return CheckQsaTopK(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunQsaTopK(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.qsa.attn", llmp::ir::Primitive(llmp::ir::Op::kAttention),
                 "QsaAttnKernel: a warp a token's KV head and share of its cells, 16-cell "
                 "cp.async gathers of K and V double-buffered, F16 m16n8k16 with F32 sums and "
                 "online softmax; QsaAttnCombineKernel (a query head a block) over the shares in "
                 "order",
                 1, ([](ConstNodes n) { return CheckQsaAttn(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunQsaAttn(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.q8_1", llmp::ir::Primitive(llmp::ir::Op::kPrepare),
                 "quantize_row_q8_1_cuda: rows padded to 512 values, once for every product", 1,
                 ([](ConstNodes n) { return CheckQuantizeQ8(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunQuantizeQ8(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.mmvq.prepared", llmp::ir::Primitive(llmp::ir::Op::kMatMul),
                 "original ggml_cuda_op_mul_mat_vec_q over a shared padded Q8_1 input; "
                 "original MMVQ selection, geometry and reduction, dense one-to-eight columns",
                 1, ([](ConstNodes n) { return CheckMmvqPrepared(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunMmvqPrepared(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.vecq", llmp::ir::Primitive(llmp::ir::Op::kMatMul),
                 "VecQKernel<type, rows, warps, glu>: GGML's vec_dot_*_q8_1 over up to 8 tokens a "
                 "weight read, one block per distinct expert and row block; SwiGLU in the kernel",
                 1, ([](ConstNodes n) { return CheckVecQ(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunVecQ(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.dsv4.route", llmp::ir::Primitive(llmp::ir::Op::kExpertSort),
                 "RouteKernel: a warp a token, sqrt(softplus), top-k by argmax rounds, normalized",
                 1, ([](ConstNodes n) { return CheckDsv4Route(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunDsv4Route(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.dsv4.combine", llmp::ir::Fused(llmp::ir::FusedKind::kMoeCombine),
                 "CombineKernel: four columns a thread, experts in order, then the shared expert",
                 1, ([](ConstNodes n) { return CheckDsv4Combine(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunDsv4Combine(launch, n[0]); }))

LLMP_GGML_RECIPE(kDsv4WeightedReduceName, llmp::ir::Fused(llmp::ir::FusedKind::kMoeCombine),
                 "WeightedReduceKernel: one column per thread, initial rounded F32 multiply "
                 "then five ascending rounded multiply/adds; no contraction or scratch",
                 1, ([](ConstNodes n) { return CheckDsv4OrderedReduce(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) {
                   return RunDsv4OrderedReduce(launch, n[0]);
                 }))

LLMP_GGML_RECIPE("llmp.dsv4.hc_mix", llmp::ir::Primitive(llmp::ir::Op::kHcPre),
                 "HcMixKernel: 64 chunks of a token, 24 dot products and the sum of squares", 1,
                 ([](ConstNodes n) { return CheckDsv4HcMix(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunDsv4HcMix(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.dsv4.hc_pre", llmp::ir::Primitive(llmp::ir::Op::kHcPre),
                 "HcPreKernel: a 1,024-thread block a token; mixes, Sinkhorn, weighted sum, "
                 "RMSNorm times the weight",
                 1, ([](ConstNodes n) { return CheckDsv4HcPre(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunDsv4HcPre(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.dsv4.compress", llmp::ir::Primitive(llmp::ir::Op::kSoftMax),
                 "CompressKernel: a block a compressed block, a thread a channel; the rows' "
                 "online softmax and weighted sum",
                 1, ([](ConstNodes n) { return CheckDsv4Compress(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunDsv4Compress(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.visible_top_k", llmp::ir::Primitive(llmp::ir::Op::kTopK),
                 "exact visible-prefix radix selection, ascending IDs and -1 padding", 1,
                 ([](ConstNodes n) { return CheckVisibleTopK(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunVisibleTopK(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.rounded_index_score", llmp::ir::Primitive(llmp::ir::Op::kIndexScore),
                 "one full-output literal -Inf fill, then unchanged LidScoreKernel<R>; "
                 "original 128MiB group boundaries, R1/2/4, 256/1024-key blocks; "
                 "half RN query, eight MMA steps and native FTZ/FMA head reduction; no scratch",
                 1, ([](ConstNodes n) { return CheckRoundedIndexScore(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) {
                   return RunRoundedIndexScore(launch, n[0]);
                 }))

LLMP_GGML_RECIPE("llmp.prefix_overwrite", llmp::ir::Primitive(llmp::ir::Op::kFill),
                 "exact integer bit-copy within clamped row prefix and literal fill outside", 1,
                 ([](ConstNodes n) { return CheckPrefixOverwrite(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunPrefixOverwrite(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.selected_mask", llmp::ir::Primitive(llmp::ir::Op::kFill),
                 "duplicate-safe shared bitset and one writer per output half", 1,
                 ([](ConstNodes n) { return CheckSelectedMask(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunSelectedMask(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.dsv4.lid_topk", llmp::ir::Primitive(llmp::ir::Op::kIndexScore),
                 "LidScoreKernel<R>: R rows' 64 heads a block on mma.sync F16 (F32 sums), 64 keys "
                 "a step; TopKKernel: a block a row, a radix select, ties to the lower row",
                 1, ([](ConstNodes n) { return CheckDsv4LidTopK(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunDsv4LidTopK(launch, n[0]); }))

LLMP_GGML_RECIPE(kFlashAttnOwnersName, llmp::ir::Primitive(llmp::ir::Op::kAttention),
                 "one to four active F16 independent K/V roots; whole-two/three, "
                 "split-four/eight/twelve or offset-filtered equal-width partial MMA partitions",
                 1, ([](ConstNodes n) { return CheckFlashAttnOwnersNode(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) -> std::expected<void, KernelFailure> {
                   auto inputs = FlashAttnOwnersFromNode(n[0]);
                   if (!inputs) return std::unexpected(inputs.error());
                   return FlashAttnOwnerRoots(launch, *inputs);
                 }))

LLMP_GGML_RECIPE(
    kCausalMaskName, llmp::ir::Primitive(llmp::ir::Op::kFill),
    "packed F16/F32 causal/ring or noncausal block mask from fresh segment I32 positions", 1,
    ([](ConstNodes n) { return CheckCausalMask(n[0]); }),
    ([](LaunchContext& launch, Nodes n) { return RunCausalMask(launch, n[0]); }))

LLMP_GGML_RECIPE("llmp.dsv4.sparse_mask", llmp::ir::Primitive(llmp::ir::Op::kFill),
                 "SparseMaskKernel: a block a row, the window's mask copied, -inf, then the "
                 "selected rows 0",
                 1, ([](ConstNodes n) { return CheckDsv4SparseMask(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunDsv4SparseMask(launch, n[0]); }))

LLMP_GGML_RECIPE(kFlashAttnVecQ8Name, llmp::ir::Primitive(llmp::ir::Op::kAttention),
                 "ggml_cuda_flash_attn_ext_vec_case<D, Q8_0, Q8_0> for D 64, 256, and 512: "
                 "flash_attn_mask_to_KV_max<ncols> from 1,024 query rows, flash_attn_ext_vec<D, "
                 "1 or 2, Q8_0, Q8_0, softcap at D 256>, flash_attn_combine_results<D> over "
                 "parallel blocks; Q to Q8_1 in shared memory; upstream launch configuration",
                 1, ([](ConstNodes n) { return CheckFlashAttnVecQuant(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return FlashAttnVecQuant(launch, n[0]); }))

LLMP_GGML_RECIPE(kFlashAttnVecQ4Name, llmp::ir::Primitive(llmp::ir::Op::kAttention),
                 "ggml_cuda_flash_attn_ext_vec_case<D, Q4_0, Q4_0> for D 64 and 256: "
                 "flash_attn_mask_to_KV_max<ncols> from 1,024 query rows, flash_attn_ext_vec<D, "
                 "1 or 2, Q4_0, Q4_0, softcap at D 256>, flash_attn_combine_results<D> over "
                 "parallel blocks; Q to Q8_1 in shared memory; upstream launch configuration",
                 1, ([](ConstNodes n) { return CheckFlashAttnVecQuant(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return FlashAttnVecQuant(launch, n[0]); }))

LLMP_GGML_RECIPE(kSetRowsQuantName, llmp::ir::Primitive(llmp::ir::Op::kCacheStore),
                 "QuantizedStores: a thread per 32-value block, k_set_rows_quant's indexing, F32 "
                 "rows at I32 or I64 indices into Q8_0 or Q4_0 by the reference quantizers' IEEE "
                 "steps; 256 threads a block, no scratch",
                 1, ([](ConstNodes n) { return CheckSetRowsQuant(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return SetRowsQuant(launch, n[0]); }))

LLMP_GGML_RECIPE(kSetRowsQuantGroupName, llmp::ir::Primitive(llmp::ir::Op::kCacheStore),
                 "QuantizedStores over two through sixteen independent stores, a grid row each; "
                 "descriptors in launch arguments; no scratch",
                 0, (&CheckSetRowsQuantGroup), (&SetRowsQuantGroup))

LLMP_GGML_RECIPE(kRopeSetRowsQuantName, llmp::ir::Fused(llmp::ir::FusedKind::kRopeStore),
                 "RopeQuantizedStore<has_ff>: a thread per 32-value block of a token's row, "
                 "rope_neox's rotation (GGML's device flags), then the reference quantizers' IEEE "
                 "steps into Q8_0 or Q4_0; 256 threads a block, no scratch",
                 2, ([](ConstNodes n) { return CheckRopeSetRowsQuant(n[0], n[1]); }),
                 ([](LaunchContext& launch, Nodes n) {
                   return RopeSetRowsQuant(launch, n[0], n[1]);
                 }))

LLMP_GGML_RECIPE("llmp.select", llmp::ir::Primitive(llmp::ir::Op::kSelect),
                 "I32 nonzero predicate; literal byte copy from chosen dense scalar branch; "
                 "broadcast without arithmetic or inspection of unselected payload",
                 1, ([](ConstNodes n) { return CheckSelect(n[0]); }),
                 ([](LaunchContext& launch, Nodes n) { return RunSelect(launch, n[0]); }))

LLMP_GGML_ACCESS_RECIPE("ggml.norm", &LayerNormAccess)
LLMP_GGML_ACCESS_RECIPE("ggml.scale", &ScaleAccess)
LLMP_GGML_ACCESS_RECIPE("ggml.unary", &UnaryAccess)
LLMP_GGML_ACCESS_RECIPE("ggml.repeat", &RepeatAccess)
LLMP_GGML_ACCESS_RECIPE("ggml.concat", &ConcatAccess)
LLMP_GGML_ACCESS_RECIPE("ggml.add", &AddAccess)
LLMP_GGML_ACCESS_RECIPE("ggml.mul", &MulAccess)
LLMP_GGML_ACCESS_RECIPE("ggml.rms_norm", &RmsAccess)
LLMP_GGML_ACCESS_RECIPE("ggml.rms_norm_mul.fused", &RmsMulAccess)
LLMP_GGML_ACCESS_RECIPE("ggml.rms_norm_mul.unfused", &RmsThenMulAccess)
LLMP_GGML_ACCESS_RECIPE(kRopeExtName, &RopeExtAccess)
LLMP_GGML_ACCESS_RECIPE("ggml.convert", &GammaConvertAccess)
LLMP_GGML_ACCESS_RECIPE("ggml.cont", &ContAccess)
LLMP_GGML_ACCESS_RECIPE(kNormRopeSectionsName, &NormRopeSectionsAccess)
LLMP_GGML_ACCESS_RECIPE(kSpatialChannelNormName, &SpatialChannelNormAccess)
LLMP_GGML_ACCESS_RECIPE("ggml.l2_norm", &L2NormAccess)
LLMP_GGML_ACCESS_RECIPE(kLayerNormModulateSplitName, &ModulateAccess)
LLMP_GGML_ACCESS_RECIPE(kGatedResidualSplitName, &GatedAccess)
LLMP_GGML_ACCESS_RECIPE(kGatedResidualModulateSplitName, &JoinedAccess)
LLMP_GGML_ACCESS_RECIPE(kHcCombineName, &HcCombineAccess)
LLMP_GGML_ACCESS_RECIPE(kGdnGatesName, &GdnGatesAccess)
LLMP_GGML_ACCESS_RECIPE(kHcPostName, &HcPostAccess)
LLMP_GGML_ACCESS_RECIPE(kDsv4HcPostNormF16Name, &HcPostNormAccess)
#ifdef LLMP_GGML_LOCAL_ACCESS_MACRO
#undef LLMP_GGML_LOCAL_ACCESS_MACRO
#undef LLMP_GGML_ACCESS_RECIPE
#endif
