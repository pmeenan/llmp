// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// One explicit row per primitive and fused key. No defaults: a new operation
// must choose its structural metadata here. Shape-dependent constraints remain
// in the inference and decomposition implementations.
// key, name, params, required/max sources, inference handler, parameter validator,
// layout, partition rule, decomposes, decomposition handler, reference path,
// direct write mask, aliased outputs. Handlers are typed in their owning modules;
// missing symbols fail compilation. NoDecomposition and kWhole are explicit.
  // Structure.
LLMP_PRIMITIVE(kInput, "input", 0, 0, 0, Leaf, ValidateNone, kStructural, kWhole, false, NoDecomposition, kDirect, 0, -1, -1)

LLMP_PRIMITIVE(kWeight, "weight", 0, 0, 0, Leaf, ValidateNone, kStructural, kWhole, false, NoDecomposition, kDirect, 0, -1, -1)

LLMP_PRIMITIVE(kState, "state", 0, 0, 0, Leaf, ValidateNone, kStructural, kWhole, false, NoDecomposition, kDirect, 0, -1, -1)

LLMP_PRIMITIVE(kView, "view", 0, 1, 1, InferView, ValidateNone, kStructural, kView, false, NoDecomposition, kDirect, 0, -1, -1)

LLMP_PRIMITIVE(kReshape, "reshape", 0, 1, 1, InferReshape, ValidateNone, kStructural, kReshape, false, NoDecomposition, kDirect, 0, -1, -1)

LLMP_PRIMITIVE(kPermute, "permute", sizeof(PermuteParams), 1, 1, InferPermute, ValidateNone, kStructural, kPermute, false, NoDecomposition, kDirect, 0, -1, -1)

LLMP_PRIMITIVE(kCont, "cont", 0, 1, 1, InferCont, ValidateNone, kFixed, kCont, false, NoDecomposition, kDirect, 0, -1, -1)

LLMP_PRIMITIVE(kConcat, "concat", sizeof(ConcatParams), 2, 2, InferConcat, ValidateNone, kConcat, kConcat, false, NoDecomposition, kDirect, 0, -1, -1)

LLMP_PRIMITIVE(kRepeat, "repeat", 0, 1, 1, InferRepeat, ValidateNone, kElementwise, kWhole, false, NoDecomposition, kDirect, 0, -1, -1)

  // a dense tensor into another encoding (a cast)
LLMP_PRIMITIVE(kConvert, "convert", 0, 1, 1, InferConvert, ValidateNone, kElementwise, kUnary, false, NoDecomposition, kDirect, 0, -1, -1)

  // rows gathered by index
LLMP_PRIMITIVE(kGetRows, "get_rows", 0, 2, 2, InferGetRows, ValidateNone, kFixed, kGetRows, false, NoDecomposition, kDirect, 0, -1, -1)
  // Products.
LLMP_PRIMITIVE(kMatMul, "mul_mat", sizeof(ProductParams), 2, 2, InferMatMul, ValidateNone, kFixed, kMatMul, false, NoDecomposition, kDirect, 0, -1, -1)

  // grouped: each row times its selected matrices
LLMP_PRIMITIVE(kMatMulId, "mul_mat_id", sizeof(ProductParams), 3, 3, InferMatMulId, ValidateNone, kFixed, kMatMulId, false, NoDecomposition, kDirect, 0, -1, -1)

  // activations into a product's input form (a quantization)
LLMP_PRIMITIVE(kPrepare, "prepare", 0, 1, 1, InferPrepare, ValidateNone, kFixed, kPrepare, false, NoDecomposition, kDirect, 0, -1, -1)

  // routed rows sorted by matrix: a grouped product's worklist
LLMP_PRIMITIVE(kExpertSort, "expert_sort", sizeof(ExpertSortParams), 1, 1, InferExpertSort, ValidateNone, kFixed, kWhole, false, NoDecomposition, kDirect, 0, -1, -1)
  // Attention and masks.
  // AttnParams
LLMP_PRIMITIVE(kAttention, "attention", sizeof(AttnParams), 3, 5, InferAttention, ValidateNone, kFixed, kAttention, true, DecomposePrimitiveAttention, kDirect, 0, -1, -1)

  // MaskParams
LLMP_PRIMITIVE(kMaskBuild, "mask_build", sizeof(MaskParams), 2, 2, InferMaskBuild, ValidateNone, kFixed, kWhole, false, NoDecomposition, kDirect, 0, -1, -1)

  // rows stored at indices (set_rows)
LLMP_PRIMITIVE(kCacheStore, "cache_store", 0, 3, 3, InferCacheStore, ValidateNone, kFixed, kCacheStore, false, NoDecomposition, kDirect, 1, -1, -1)

  // sparse attention's cell scores
LLMP_PRIMITIVE(kIndexScore, "index_score", 0, 3, 4, InferIndexScore, ValidateNone, kFixed, kWhole, true, DecomposePrimitiveIndexScore, kDirect, 0, -1, -1)

  // compressed KV pooling
LLMP_PRIMITIVE(kPool, "pool", sizeof(PoolParams), 1, 2, InferPool, ValidateNone, kFixed, kWhole, true, DecomposePrimitivePool, kDirect, 0, -1, -1)
  // Norms and rotations.
LLMP_PRIMITIVE(kRmsNorm, "rms_norm", sizeof(NormParams), 1, 1, InferNorm, ValidateNone, kFixed, kRows, false, NoDecomposition, kDirect, 0, -1, -1)

LLMP_PRIMITIVE(kL2Norm, "l2_norm", sizeof(NormParams), 1, 1, InferNorm, ValidateNone, kFixed, kRows, false, NoDecomposition, kDirect, 0, -1, -1)

LLMP_PRIMITIVE(kLayerNorm, "layer_norm", sizeof(NormParams), 1, 1, InferNorm, ValidateNone, kFixed, kRows, false, NoDecomposition, kDirect, 0, -1, -1)

LLMP_PRIMITIVE(kChannelNorm, "channel_norm", sizeof(NormParams), 1, 1, InferNorm, ValidateNone, kFixed, kChannelNorm, false, NoDecomposition, kDirect, 0, -1, -1)

  // RopeParams
LLMP_PRIMITIVE(kRope, "rope", sizeof(RopeParams), 2, 3, InferRope, ValidateNone, kFixed, kRope, false, NoDecomposition, kDirect, 0, -1, -1)
  // Elementwise.
LLMP_PRIMITIVE(kAdd, "add", 0, 2, 2, InferBinary, ValidateNone, kElementwise, kAdditive, false, NoDecomposition, kDirect, 0, -1, -1)

LLMP_PRIMITIVE(kSub, "sub", 0, 2, 2, InferBinary, ValidateNone, kElementwise, kAdditive, false, NoDecomposition, kDirect, 0, -1, -1)

LLMP_PRIMITIVE(kMul, "mul", 0, 2, 2, InferBinary, ValidateNone, kElementwise, kMultiply, false, NoDecomposition, kDirect, 0, -1, -1)

LLMP_PRIMITIVE(kDiv, "div", 0, 2, 2, InferBinary, ValidateNone, kElementwise, kDivide, false, NoDecomposition, kDirect, 0, -1, -1)

  // x · scale + bias
LLMP_PRIMITIVE(kScale, "scale", sizeof(ScaleParams), 1, 1, InferScale, ValidateNone, kElementwise, kScale, false, NoDecomposition, kDirect, 0, -1, -1)

  // UnaryParams
LLMP_PRIMITIVE(kUnary, "unary", sizeof(UnaryParams), 1, 1, InferUnary, ValidateNone, kElementwise, kUnary, false, NoDecomposition, kDirect, 0, -1, -1)

LLMP_PRIMITIVE(kClamp, "clamp", sizeof(ClampParams), 1, 1, InferClamp, ValidateNone, kElementwise, kUnary, false, NoDecomposition, kDirect, 0, -1, -1)

LLMP_PRIMITIVE(kFill, "fill", sizeof(FillParams), 0, 0, InferFill, ValidateNone, kFixed, kWhole, false, NoDecomposition, kDirect, 0, -1, -1)

  // GluParams
LLMP_PRIMITIVE(kGlu, "glu", sizeof(GluParams), 1, 2, InferGlu, ValidateNone, kElementwise, kGlu, false, NoDecomposition, kDirect, 0, -1, -1)

LLMP_PRIMITIVE(kSoftcap, "softcap", sizeof(SoftcapParams), 1, 1, InferSoftcap, ValidateNone, kFixed, kUnary, false, NoDecomposition, kDirect, 0, -1, -1)
  // Reductions and selection.
LLMP_PRIMITIVE(kSumRows, "sum_rows", 0, 1, 1, InferSumRows, ValidateNone, kFixed, kRowReduce, false, NoDecomposition, kDirect, 0, -1, -1)

LLMP_PRIMITIVE(kSoftMax, "soft_max", sizeof(SoftMaxParams), 1, 3, InferSoftMax, ValidateNone, kFixed, kSoftMax, false, NoDecomposition, kDirect, 0, -1, -1)

LLMP_PRIMITIVE(kArgmax, "argmax", sizeof(ArgmaxParams), 1, 1, InferArgmax, ValidateNone, kFixed, kArgmax, false, NoDecomposition, kDirect, 0, -1, -1)

LLMP_PRIMITIVE(kArgsort, "argsort", sizeof(ArgsortParams), 1, 1, InferArgsort, ValidateNone, kFixed, kRowReduce, false, NoDecomposition, kDirect, 0, -1, -1)

LLMP_PRIMITIVE(kTopK, "top_k", sizeof(TopKParams), 1, 1, InferTopK, ValidateNone, kFixed, kRowReduce, false, NoDecomposition, kDirect, 0, -1, -1)
  // Routing.
  // RouteParams: scores into selected experts and weights
LLMP_PRIMITIVE(kRoute, "route", sizeof(RouteParams), 1, 4, InferRoute, ValidateNone, kFixed, kWhole, true, DecomposePrimitiveRoute, kDirect, 0, -1, -1)

  // the selected experts' outputs, weighted and summed
LLMP_PRIMITIVE(kExpertReduce, "expert_reduce", 0, 2, 4, InferExpertReduce, ValidateNone, kFixed, kExpertReduce, true, DecomposePrimitiveExpertReduce, kDirect, 0, -1, -1)
  // Recurrences and convolutions.
LLMP_PRIMITIVE(kSsmConv, "ssm_conv", 0, 2, 2, InferSsmConv, ValidateNone, kFixed, kWhole, true, DecomposePrimitiveSsmConv, kDirect, 0, -1, -1)

LLMP_PRIMITIVE(kGatedDeltaNet, "gated_delta_net", 0, 6, 6, InferGatedDeltaNet, ValidateNone, kFixed, kWhole, true, DecomposePrimitiveGatedDeltaNet, kDirect, 0, -1, -1)

LLMP_PRIMITIVE(kConv2d, "conv_2d", sizeof(Conv2dParams), 2, 3, InferConv2d, ValidateNone, kFixed, kWhole, false, NoDecomposition, kDirect, 0, -1, -1)

LLMP_PRIMITIVE(kUpsample, "upsample", sizeof(UpsampleParams), 1, 1, InferUpsample, ValidateNone, kFixed, kWhole, false, NoDecomposition, kDirect, 0, -1, -1)
  // Hyper-connections: composite ops with a mandatory primitive
  // decomposition.
  // streams folded into one
LLMP_PRIMITIVE(kHcPre, "hc_pre", sizeof(HcParams), 2, 2, InferHcPre, ValidateNone, kFixed, kWhole, true, DecomposePrimitiveHcPre, kDirect, 0, -1, -1)

  // one output spread back over the streams
LLMP_PRIMITIVE(kHcPost, "hc_post", sizeof(HcParams), 3, 4, InferHcPost, ValidateNone, kFixed, kWhole, true, DecomposePrimitiveHcPost, kDirect, 0, -1, -1)

  // HcParams: the mixing weights
LLMP_PRIMITIVE(kHcMix, "hc_mix", sizeof(HcParams), 3, 3, InferHcMix, ValidateNone, kFixed, kWhole, true, DecomposePrimitiveHcMix, kDirect, 0, -1, -1)
  // Sampling and collectives.
LLMP_PRIMITIVE(kSample, "sample", sizeof(SampleParams), 1, 2, InferSample, ValidateNone, kFixed, kWhole, false, NoDecomposition, kDirect, 0, -1, -1)

LLMP_PRIMITIVE(kAllReduce, "all_reduce", sizeof(CollectiveParams), 1, 1, InferCollectiveCopy, ValidateNone, kFixed, kWhole, false, NoDecomposition, kCollective, 0, -1, -1)

LLMP_PRIMITIVE(kAllGather, "all_gather", sizeof(CollectiveParams), 1, 1, InferAllGather, ValidateNone, kFixed, kWhole, false, NoDecomposition, kCollective, 0, -1, -1)

LLMP_PRIMITIVE(kAllToAll, "all_to_all", sizeof(CollectiveParams), 1, 1, InferCollectiveCopy, ValidateNone, kFixed, kWhole, false, NoDecomposition, kCollective, 0, -1, -1)

LLMP_PRIMITIVE(kSend, "send", sizeof(CollectiveParams), 1, 1, InferSend, ValidateNone, kFixed, kWhole, false, NoDecomposition, kCollective, 0, -1, -1)

LLMP_PRIMITIVE(kRecv, "recv", sizeof(CollectiveParams), 0, 0, InferRecv, ValidateNone, kFixed, kWhole, false, NoDecomposition, kCollective, 0, -1, -1)
  // A pattern's result; FusedKind names it.
LLMP_PRIMITIVE(kFused, "fused", 0, 0, 0, InferInvalid, ValidateNone, kFixed, kWhole, false, NoDecomposition, kDecomposition, 0, -1, -1)
  // Exact visibility-bounded selection and its bit-preserving mask.
LLMP_PRIMITIVE(kVisibleTopK, "visible_top_k", sizeof(TopKParams), 2, 2, InferVisibleTopK, ValidateNone, kFixed, kWhole, true, DecomposePrimitiveVisibleTopK, kDirect, 0, -1, -1)

LLMP_PRIMITIVE(kSelectedMask, "selected_mask", sizeof(SelectedMaskParams), 2, 2, InferSelectedMask, ValidateNone, kFixed, kWhole, true, DecomposePrimitiveSelectedMask, kDirect, 0, -1, -1)

LLMP_PRIMITIVE(kRoundedIndexScore, "rounded_index_score", sizeof(RoundedIndexScoreParams), 4, 4, InferRoundedIndexScore, ValidateNone, kFixed, kWhole, true, DecomposePrimitiveRoundedIndexScore, kDirect, 0, -1, -1)

LLMP_PRIMITIVE(kPrefixOverwrite, "prefix_overwrite", sizeof(PrefixOverwriteParams), 2, 2, InferPrefixOverwrite, ValidateNone, kFixed, kWhole, false, NoDecomposition, kDirect, 0, -1, -1)

LLMP_PRIMITIVE(kSelect, "select", 0, 3, 3, InferSelect, ValidateNone, kFixed, kWhole, false, NoDecomposition, kDirect, 0, -1, -1)

LLMP_FUSED(kNone, "none", 0, 0, 0, InferInvalid, ValidateNone, kFixed, kWhole, false, NoDecomposition, kNone, 0, -1, -1)

  // (x, w): NormParams: rms_norm(x) · w
LLMP_FUSED(kRmsNormMul, "rms_norm_mul", sizeof(NormParams), 2, 2, InferFused, ValidateNone, kFixed, kWhole, true, DecomposeRmsNormMul, kDecomposition, 0, -1, -1)
  // (x [D, H, T], pos [T]) or (x [D, H, T], w [D], pos [T], freq?):
  // NormRopeParams: rope(rms_norm(x, eps) [· w]).
LLMP_FUSED(kRmsNormMulRope, "rms_norm_mul_rope", sizeof(NormRopeParams), 2, 4, InferFused, ValidateNormRope, kFixed, kWhole, true, DecomposeRmsNormMulRope, kDecomposition, 0, -1, -1)

  // (x, w, r): NormParams: rms_norm(x) · w + r
LLMP_FUSED(kRmsNormMulAdd, "rms_norm_mul_add", sizeof(NormParams), 3, 3, InferFused, ValidateNone, kFixed, kWhole, true, DecomposeRmsNormMulAdd, kDecomposition, 0, -1, -1)
  // (x [D, H, T], pos, cache [D · H, C], idx [T], freq?): RopeParams: the
  // rotated rows stored; the output is a view of cache.
LLMP_FUSED(kRopeStore, "rope_store", sizeof(RopeParams), 4, 5, InferFused, ValidateRope, kFixed, kWhole, true, DecomposeRopeStore, kDecomposition, 0, 2, -1)
  // (w_gate, w_up, x): GluParams: glu(w_gate · x, w_up · x), the gate's
  // product the GLU's gate (GGML's ggml_glu_split(gate, up) over its
  // mul_mat(w_gate, x) and mul_mat(w_up, x); the weights first, as
  // kMatMulAdd's).
LLMP_FUSED(kMatMulGlu, "mul_mat_glu", sizeof(GluParams), 3, 3, InferFused, ValidateFlag1Reserved2, kFixed, kWhole, true, DecomposeMatMulGlu, kDecomposition, 0, -1, -1)

  // (w, x, b): w · x + b
LLMP_FUSED(kMatMulAdd, "mul_mat_add", 0, 3, 3, InferFused, ValidateNone, kFixed, kWhole, true, DecomposeMatMulAdd, kDecomposition, 0, -1, -1)
  // (q [D, R, H, owners], mask [C, >= R, 1, owners], k0, ..., v0, ...):
  // OwnerAttnParams: joined requests' rows, each owner's R rows attending to
  // its own cache [D, C_o, Hk, 1] (C_o <= C): owner o's queries q[:, :, :,
  // o] and its mask mask[:C_o, :R, 0, o] (the first C_o cells and R rows
  // of its own matrix, so each owner keeps its mask's row padding);
  // [Dv, H, R, owners], owner o's attention at o. One to four owners.
  // Its cohort and offset (amended 2026-10-09) place the node among a
  // step's joined owners when they run as several nodes of up to four
  // (GGML's four-root carriers): the cohort is every joined owner, the
  // offset the node's first one there. Like a product's accumulation, they
  // are hashed choices that do not change the definition (an
  // implementation partitions its grid by them); 0 and 0 is a lone node.
LLMP_FUSED(kOwnerAttention, "owner_attention", sizeof(OwnerAttnParams), 4, 10, InferFused, ValidateAttention, kFixed, kWhole, true, DecomposeOwnerAttention, kDecomposition, 0, -1, -1)
  // A batch key, never a node (the builder and verifier refuse one, and it
  // has no decomposition): consecutive independent kCacheStore nodes that
  // one implementation runs as one launch (planner/pattern.h
  // grouped_stores). A capability of it describes one member store
  // (cache, src, idx) and declares its batch bound (execution/registry.h
  // Capability::batch). Its first form, a node of two or three stores,
  // held none of GGML's groups (two to sixteen) and was dropped
  // (2026-10-09).
LLMP_FUSED(kGroupedStores, "grouped_stores", 0, 0, 0, InferBatchKey, ValidateNone, kFixed, kWhole, false, NoDecomposition, kNone, 0, -1, -1)
  // (gate [n, U, T], up [n, U, T], ids [U, T], s_gate [E], s_up [E]):
  // GluParams: glu(gate · s_gate[ids], up · s_up[ids]).
LLMP_FUSED(kMoeGlu, "moe_glu", sizeof(GluParams), 5, 5, InferFused, ValidateFlag1Reserved2, kFixed, kWhole, true, DecomposeMoeGlu, kDecomposition, 0, -1, -1)
  // (down [n, U, T], ids, weights [U, T], shared [n, T], shared_gate? [1, T],
  // s_down? [E]): expert_reduce(down, weights, ids, s_down) + shared ·
  // sigmoid(shared_gate), or + shared where the shared expert has no gate
  // (DeepSeek V4's).
LLMP_FUSED(kMoeCombine, "moe_combine", 0, 4, 6, InferFused, ValidateNone, kFixed, kWhole, true, DecomposeMoeCombine, kDecomposition, 0, -1, -1)
  // (x [W, hc, T], w [W · hc]): NormParams: each stream's rms_norm (over W,
  // Qwen3.8's llmp.hc.norm), then as [W · hc, T], · w. DeepSeek V4's flat
  // norm over all W · hc is kHcNormPost's second output, or rms_norm of the
  // reshape.
LLMP_FUSED(kHcNorm, "hc_norm", sizeof(NormParams), 2, 2, InferFused, ValidateNone, kFixed, kWhole, true, DecomposeHcNorm, kDecomposition, 0, -1, -1)
  // (y, x, post, comb?): HcNormPostParams: two outputs, hc_post's p and
  // rms_norm(p as [W · hc, T]).
LLMP_FUSED(kHcNormPost, "hc_norm_post", sizeof(HcNormPostParams), 3, 4, InferFused, ValidateNone, kFixed, kWhole, true, DecomposeHcNormPost, kDecomposition, 0, -1, -1)
  // (q, wq, k, wk, pos, freq?): NormRopeParams: two outputs, each
  // rope(rms_norm(·) · w).
LLMP_FUSED(kQkPrep, "qk_prep", sizeof(NormRopeParams), 5, 6, InferFused, ValidateNormRope, kFixed, kWhole, true, DecomposeQkPrep, kDecomposition, 0, -1, -1)
  // (w0, w1, ..., x): an output per product wi · x, up to four; the
  // weights first, as kMatMulGlu's and kMatMulAdd's.
LLMP_FUSED(kPreparedProducts, "prepared_products", 0, 2, 5, InferFused, ValidateNone, kFixed, kWhole, true, DecomposePreparedProducts, kDecomposition, 0, -1, -1)

  // (x, w, z): NormGateParams: rms_norm(x) · w · sigmoid(z) or SiLU(z)
LLMP_FUSED(kNormGate, "norm_gate", sizeof(NormGateParams), 3, 3, InferFused, ValidateNone, kFixed, kWhole, true, DecomposeNormGate, kDecomposition, 0, -1, -1)

  // (sample, noise, dt [1]): sample + dt · noise
LLMP_FUSED(kEulerStep, "euler_step", 0, 3, 3, InferFused, ValidateNone, kFixed, kWhole, true, DecomposeEulerStep, kDecomposition, 0, -1, -1)
  // (x [2W, 2H, Co], shortcut [W, H, Ci]): DupUpParams: x plus Qwen-Image's
  // DupUp3D of the shortcut at its last kept frame (kernels/ggml/ops_ext.h
  // DupUpAddF32): output channel o's sub-pixel (b, d) reads input channel
  // (((o · ft + ft - 1) · 2 + b) · 2 + d) / (Co · ft · 4 / Ci).
LLMP_FUSED(kDupUpAdd, "dup_up_add", sizeof(DupUpParams), 2, 2, InferFused, ValidateNone, kFixed, kWhole, true, DecomposeDupUpAdd, kDecomposition, 0, -1, -1)
  // (x [W, hc, T], norm [W · hc], inject? [W · hc, hc], out? [W, T],
  // logits? [hc, T]): HcNormPostParams: the streams s = x, or with out and
  // logits x combined with the last block's output first (kHcCombine); the
  // normed streams xn = kHcNorm(s, norm), and with inject the next combine's
  // logits inject · xn (from xn in F32). Outputs: xn, then the logits (with
  // inject), then s (with out).
LLMP_FUSED(kHcPrep, "hc_prep", sizeof(HcNormPostParams), 2, 5, InferFused, ValidateNone, kFixed, kWhole, true, DecomposeHcPrep, kDecomposition, 0, -1, -1)
  // (x): ScaledUnaryParams: fn(x · scale).
LLMP_FUSED(kScaledUnary, "scaled_unary", sizeof(ScaledUnaryParams), 1, 1, InferFused, ValidateScaledUnary, kFixed, kWhole, true, DecomposeScaledUnary, kDecomposition, 0, -1, -1)
  // (xn [W · hc, T], gate [W · hc, T]) or (x [W, hc, T], norm [W · hc],
  // gate [W · hc, T]): HcFoldParams: (1 / hc) Σ_k xn_k · sigmoid(gate_k)
  // over the streams in order (kHcPre's gated fold of xn as [W, hc, T], or
  // of kHcNorm(x, norm, eps)), [W, T]; with `prepared` also the result
  // prepared into MXFP8 rows (kPrepare's), with `rounded` also rounded to
  // BF16, in that order after it.
LLMP_FUSED(kHcFold, "hc_fold", sizeof(HcFoldParams), 2, 3, InferFused, ValidateHcFold, kFixed, kWhole, true, DecomposeHcFold, kDecomposition, 0, -1, -1)
  // (x [W, hc, T], out [W, T], logits [hc, T]): HcParams: x + out ·
  // 2 · sigmoid(logits / hc) on each stream (hc.combine's).
LLMP_FUSED(kHcCombine, "hc_combine", sizeof(HcParams), 3, 3, InferFused, ValidateHcCombine, kFixed, kWhole, true, DecomposeHcCombine, kDecomposition, 0, -1, -1)
  // (rows [C, T], history [(K - 1) · C], w [K, C]): GdnConvParams: the
  // causal depthwise convolution of the rows after the history (tap j of
  // channel c at c · (K - 1) + j) and SiLU, [C, T]; its first qk_channels
  // channels, heads of `head`, L2-normed as rms_norm(x, eps) · scale.
LLMP_FUSED(kGdnConv, "gdn_conv", sizeof(GdnConvParams), 3, 3, InferFused, ValidateNone, kFixed, kWhole, true, DecomposeGdnConv, kDecomposition, 0, -1, -1)
  // (rows [C, T], history? [(K - 1) · C]): GdnHistoryParams: the
  // convolution's next history, [taps · C, 1]: the last `taps` of the
  // history (when given) followed by the rows, each channel's taps
  // together.
LLMP_FUSED(kGdnHistory, "gdn_history", sizeof(GdnHistoryParams), 1, 2, InferFused, ValidateNone, kFixed, kWhole, true, DecomposeGdnHistory, kDecomposition, 0, -1, -1)
  // (q, k, v, g, beta, state [S], idx? [1]): GdnStepParams: kGatedDeltaNet
  // over state viewed as [Sk, Sv, H, 1]: its attention, and with `write`
  // the new state stored over state at idx (an output that is a view of
  // state).
LLMP_FUSED(kGdnStep, "gdn_step", sizeof(GdnStepParams), 6, 7, InferFused, ValidateGdnStep, kFixed, kWhole, true, DecomposeGdnStep, kDecomposition, 0, -1, 5)
  // (alpha [H, T], beta [H, T], dt_bias [H], a [H]): two outputs, the
  // decay's log softplus(alpha + dt_bias) · a and sigmoid(beta), each [1,
  // H, T, 1].
LLMP_FUSED(kGdnGates, "gdn_gates", 0, 4, 4, InferFused, ValidateNone, kFixed, kWhole, true, DecomposeGdnGates, kDecomposition, 0, -1, -1)
  // (attn [d · H, T], q [2 · d · H, T]): HeadGateParams: attn times
  // sigmoid of each head's gate (q's second d of each head's 2 · d).
LLMP_FUSED(kHeadGate, "head_gate", sizeof(HeadGateParams), 2, 2, InferFused, ValidateNone, kFixed, kWhole, true, DecomposeHeadGate, kDecomposition, 0, -1, -1)
  // (raw [D, C], blocks [D, N], w [D], block_pos [4 · B], cells [B],
  // positions? [4 · T]): BlockPoolParams: the block keys of the first B
  // blocks of `ratio` raw rows, each their mean, rms_norm · w, rotated at
  // its block's position and rounded to BF16, stored in blocks at cells:
  // a view of blocks. An implementation may write only the blocks the
  // step's positions complete (the others hold what earlier steps stored):
  // a block's key never changes once its rows are written.
LLMP_FUSED(kMeanPoolNormRopeStore, "mean_pool_norm_rope_store", sizeof(BlockPoolParams), 5, 6, InferFused, ValidatePoolRope, kFixed, kWhole, true, DecomposeMeanPoolNormRopeStore, kDecomposition, 0, 1, -1)
  // (q [d, H, T], k [d, C, Hk], v [d, C, Hk], qi [Di, Hi, T], blocks [Di,
  // B], causal [C, T] F32, seen [B, T] F32, mask [C, >= T], positions? [4 ·
  // T]): BlockSparseAttnParams: block-sparse selection and attention:
  // each block's score Σ_h relu(bf16(qi_h) · block), plus 1e9 where `seen`
  // hides the block's last position, spread over its `ratio` cells, plus
  // `causal`; the `width` best cells kept; attention of q over k and v
  // under `mask` and -inf but at the kept cells. [d · H, T].
LLMP_FUSED(kBlockSparseAttention, "block_sparse_attention", sizeof(BlockSparseAttnParams), 8, 9, InferFused, ValidateAttention, kFixed, kWhole, true, DecomposeBlockSparseAttention, kDecomposition, 0, -1, -1)
  // (logits [E, T], x [W, T], gate [W]): RouteParams: kRoute's ids and
  // weights, and the shared expert's gate logit Σ x · gate [1, T].
LLMP_FUSED(kMoeRouter, "moe_router", sizeof(RouteParams), 3, 3, InferFused, ValidateFlag1Reserved2, kFixed, kWhole, true, DecomposeMoeRouter, kDecomposition, 0, -1, -1)
  // (gate_up [W, 2F, E], down [F, W, E], x [W, T], ids [U, T], weights [U,
  // T], shared [W, T], gate [1, T], s_gate [E], s_up [E], s_down [E]):
  // MoeExpertsParams: the routed experts (gate_up's halves, each by its
  // expert's scale, the GLU, down) reduced with s_down and the weights,
  // plus shared · sigmoid(gate): kMoeCombine over them.
LLMP_FUSED(kMoeExperts, "moe_experts", sizeof(MoeExpertsParams), 7, 10, InferFused, ValidateFlag1Reserved2, kFixed, kWhole, true, DecomposeMoeExperts, kDecomposition, 0, -1, -1)
  // (gate [W, F, E], up [W, F, E], x [W, T], ids [U, T]): MoeExpertsParams:
  // glu(gate[ids] · x, up[ids] · x), [F, U, T].
LLMP_FUSED(kMatMulIdGlu, "mul_mat_id_glu", sizeof(MoeExpertsParams), 4, 4, InferFused, ValidateFlag1Reserved2, kFixed, kWhole, true, DecomposeMatMulIdGlu, kDecomposition, 0, -1, -1)
  // (state_kv [C, S], state_score [C, S], kv [C, T], score [C, T],
  // read_idxs [ratio * B * (overlap ? 2 : 1)] I32): CompressorParams:
  // each block's rows from [state | current] softmax pooled per channel.
  // Without overlap C = D; with overlap C = 2D, each block takes its
  // previous rows' first D channels, then its current rows' second D.
  // Overlap's source also has a final zero row with score -inf. The
  // indices address this source; [D, 1, B] F32 before norm or rotation.
LLMP_FUSED(kSoftmaxPoolCompress, "softmax_pool_compress", sizeof(CompressorParams), 5, 5, InferFused, ValidateCompressor, kFixed, kWhole, true, DecomposeSoftmaxPoolCompress, kDecomposition, 0, -1, -1)
  // (x [W, hc, T], fn [W * hc, (2 + hc) * hc], scale [3],
  // base [(2 + hc) * hc], norm [W]): SinkhornPreParams: the full-carry
  // Sinkhorn block mix. Outputs are the folded stream's weighted RMSNorm
  // [W, T], post [hc, T], and combination [hc, hc, T], all packed F32.
LLMP_FUSED(kSinkhornHcPre, "sinkhorn_hc_pre", sizeof(SinkhornPreParams), 5, 5, InferFused, ValidateNone, kFixed, kWhole, true, DecomposeSinkhornHcPre, kDecomposition, 0, -1, -1)
  // (q [D, T, H], raw [D, 1, A], compressed [D, 1, B], mask?, sinks?):
  // AttnParams, unchanged: Attention(q, permute(concat(raw, compressed, 2)),
  // the same K as V, mask, sinks). The formula needs no layout fact;
  // only zero-copy implementations require a proven contiguous state span.
LLMP_FUSED(kStateBankAttention, "state_bank_attention", sizeof(AttnParams), 3, 5, InferFused, ValidateAttention, kFixed, kWhole, true, DecomposeStateBankAttention, kDecomposition, 0, -1, -1)
  // (q [D,H,T], k [D,1,C], weights [H,T], visible I32 [T], raw F16 [W,T]):
  // RoundedIndexMaskParams. RoundedIndexScore -> VisibleTopK -> SelectedMask,
  // with the same visible operand throughout. Neither score nor selected IDs
  // escape this formula. It does not fuse generic IndexScore/TopK semantics.
LLMP_FUSED(kRoundedIndexMask, "rounded_index_mask", sizeof(RoundedIndexMaskParams), 5, 5, InferFused, ValidateNone, kFixed, kWhole, true, DecomposeRoundedIndexMask, kDecomposition, 0, -1, -1)
  // (w [K,N,G], x [K,G,T]): ProductParams. Each group uses its own
  // matrix on its input rows; packed F32 [N,G,T]. The decomposition
  // permutes x to [K,T,G], multiplies, permutes back and makes contiguous.
LLMP_FUSED(kGroupedMatMul, "grouped_mul_mat", sizeof(ProductParams), 2, 2, InferFused, ValidateProduct, kFixed, kWhole, true, DecomposeGroupedMatMul, kDecomposition, 0, -1, -1)
  // (out_a, heads, positions): RopeParams: inverse tail RoPE and staged
  // grouped output-A matrix product.
LLMP_FUSED(kInverseRopeGroupedMatMul, "inverse_rope_grouped_mul_mat", sizeof(RopeParams), 3, 3, InferFused, ValidateRope, kFixed, kWhole, true, DecomposeInverseRopeGroupedMatMul, kDecomposition, 0, -1, -1)
  // (x [W, H, C, N], gamma [C]): SpatialNormParams: each pixel's channels
  // L2-normed (over C, eps the norm's floor), times scale and gamma, and
  // with silu SiLU: the decomposition takes x to channel rows ([C, W, H,
  // N]), normalizes and returns to [W, H, C, N].
LLMP_FUSED(kSpatialChannelNorm, "spatial_channel_norm", sizeof(SpatialNormParams), 2, 2, InferFused, ValidateSpatialNorm, kFixed, kWhole, true, DecomposeSpatialChannelNorm, kDecomposition, 0, -1, -1)
  // (x [D, heads, rows], w [D], pos0 [rows], pos1, pos2): RopeSectionsParams:
  // the RMS norm of each head times w (any float encoding, read as F32),
  // then `sections` plain rotations, each of its own I32 positions over
  // the next dims[i] values of the head (offsets 0, dims[0], dims[0] +
  // dims[1]), together the whole head: a multi-axis rotation.
LLMP_FUSED(kRmsNormMulRopeSections, "rms_norm_mul_rope_sections", sizeof(RopeSectionsParams), 3, 5, InferFused, ValidateRopeSections, kFixed, kWhole, true, DecomposeRmsNormMulRopeSections, kDecomposition, 0, -1, -1)
  // (mask F32 [C, R]): MaskPackParams: the mask in F16, padded with -inf to
  // [cells, rows] (cells >= C, rows >= R): an attention's mask prepared
  // once for every attention of a step that reads it. (An implementation
  // may mark a square mask's whole rows, -0 at their first column, for an
  // attention whose keys are exactly its columns: kQkAttention's.)
LLMP_FUSED(kAttentionMaskPack, "attention_mask_pack", sizeof(MaskPackParams), 1, 1, InferFused, ValidateNone, kFixed, kWhole, true, DecomposeAttentionMaskPack, kDecomposition, 0, -1, -1)
  // (pos0 [R], pos1, pos2): RopeSectionsParams (normal pairs; eps unused):
  // F32 [2, D / 2, R], each pair's cos and sin of the sections' angles
  // (kRmsNormMulRopeSections' rotation of (1, 0)): a rotation table.
LLMP_FUSED(kRopeTable, "rope_table", sizeof(RopeSectionsParams), 1, 3, InferFused, ValidateRopeSections, kFixed, kWhole, true, DecomposeRopeTable, kDecomposition, 0, -1, -1)
  // (q [D · H, R], wq [D], k [D · Hk, R], v [D · Hk, R], mask [cells, >= R],
  // table F32 [2, D / 2, R]): QkAttentionParams: attention of q's heads,
  // each RMS-normed (eps), times wq and rotated by the table (normal
  // pairs), over k's and v's heads (head h reads h / (H / Hk)) and the
  // mask's first R rows and columns, scaled: [D · H, R], each row's heads
  // in order (the channel rows a projection reads).
LLMP_FUSED(kQkAttention, "qk_attention", sizeof(QkAttentionParams), 6, 6, InferFused, ValidateNone, kFixed, kWhole, true, DecomposeQkAttention, kDecomposition, 0, -1, -1)
  // (x [C, R], s [C]) or (x [C, R], a [C], b [C]): SplitModulateParams:
  // LayerNorm(x) · (1 + s), where s is unsplit (2 sources) or row r is a
  // where r < split, else b (3 sources).
LLMP_FUSED(kLayerNormModulateSplit, "layer_norm_modulate_split", sizeof(SplitModulateParams), 2, 3, InferFused, ValidateNone, kFixed, kWhole, true, DecomposeSplitModulate<false>, kDecomposition, 0, -1, -1)
  // (carry [C, R], branch [C, R], g [C]) or (carry [C, R], branch [C, R],
  // a [C], b [C]): SplitModulateParams (eps unused): carry + branch · tanh(g),
  // where g is unsplit (3 sources) or row r is a where r < split, else b.
LLMP_FUSED(kGatedResidualSplit, "gated_residual_split", sizeof(SplitModulateParams), 3, 4, InferFused, ValidateNone, kFixed, kWhole, true, DecomposeSplitModulate<true>, kDecomposition, 0, -1, -1)
  // (carry [C, R], branch [C, R], g [C], m [C]) or (carry [C, R], branch
  // [C, R], ga [C], gb [C], ma [C], mb [C]): SplitModulateParams:
  // x = kGatedResidualSplit, then kLayerNormModulateSplit(x) (eps):
  // [x, modulated].
LLMP_FUSED(kGatedResidualModulateSplit, "gated_residual_modulate_split", sizeof(SplitModulateParams), 4, 6, InferFused, ValidateNone, kFixed, kWhole, true, DecomposeGatedResidualModulateSplit, kDecomposition, 0, -1, -1)
