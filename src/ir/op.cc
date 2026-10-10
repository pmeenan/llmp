// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "ir/op.h"

#include <string>
#include <string_view>

namespace llmp::ir {

std::string_view OpName(Op op) {
  switch (op) {
    case Op::kInput:
      return "input";
    case Op::kWeight:
      return "weight";
    case Op::kState:
      return "state";
    case Op::kView:
      return "view";
    case Op::kReshape:
      return "reshape";
    case Op::kPermute:
      return "permute";
    case Op::kCont:
      return "cont";
    case Op::kConcat:
      return "concat";
    case Op::kRepeat:
      return "repeat";
    case Op::kConvert:
      return "convert";
    case Op::kGetRows:
      return "get_rows";
    case Op::kMatMul:
      return "mul_mat";
    case Op::kMatMulId:
      return "mul_mat_id";
    case Op::kPrepare:
      return "prepare";
    case Op::kExpertSort:
      return "expert_sort";
    case Op::kAttention:
      return "attention";
    case Op::kMaskBuild:
      return "mask_build";
    case Op::kCacheStore:
      return "cache_store";
    case Op::kIndexScore:
      return "index_score";
    case Op::kPool:
      return "pool";
    case Op::kRmsNorm:
      return "rms_norm";
    case Op::kL2Norm:
      return "l2_norm";
    case Op::kLayerNorm:
      return "layer_norm";
    case Op::kChannelNorm:
      return "channel_norm";
    case Op::kRope:
      return "rope";
    case Op::kAdd:
      return "add";
    case Op::kSub:
      return "sub";
    case Op::kMul:
      return "mul";
    case Op::kDiv:
      return "div";
    case Op::kScale:
      return "scale";
    case Op::kUnary:
      return "unary";
    case Op::kClamp:
      return "clamp";
    case Op::kFill:
      return "fill";
    case Op::kGlu:
      return "glu";
    case Op::kSoftcap:
      return "softcap";
    case Op::kSumRows:
      return "sum_rows";
    case Op::kSoftMax:
      return "soft_max";
    case Op::kArgmax:
      return "argmax";
    case Op::kArgsort:
      return "argsort";
    case Op::kTopK:
      return "top_k";
    case Op::kVisibleTopK:
      return "visible_top_k";
    case Op::kSelectedMask:
      return "selected_mask";
    case Op::kRoundedIndexScore:
      return "rounded_index_score";
    case Op::kPrefixOverwrite:
      return "prefix_overwrite";
    case Op::kRoute:
      return "route";
    case Op::kExpertReduce:
      return "expert_reduce";
    case Op::kSsmConv:
      return "ssm_conv";
    case Op::kGatedDeltaNet:
      return "gated_delta_net";
    case Op::kConv2d:
      return "conv_2d";
    case Op::kUpsample:
      return "upsample";
    case Op::kHcPre:
      return "hc_pre";
    case Op::kHcPost:
      return "hc_post";
    case Op::kHcMix:
      return "hc_mix";
    case Op::kSample:
      return "sample";
    case Op::kAllReduce:
      return "all_reduce";
    case Op::kAllGather:
      return "all_gather";
    case Op::kAllToAll:
      return "all_to_all";
    case Op::kSend:
      return "send";
    case Op::kRecv:
      return "recv";
    case Op::kFused:
      return "fused";
  }
  return "unknown";
}

std::string_view FusedKindName(FusedKind kind) {
  switch (kind) {
    case FusedKind::kNone:
      return "none";
    case FusedKind::kRmsNormMul:
      return "rms_norm_mul";
    case FusedKind::kRmsNormMulRope:
      return "rms_norm_mul_rope";
    case FusedKind::kRmsNormMulAdd:
      return "rms_norm_mul_add";
    case FusedKind::kRopeStore:
      return "rope_store";
    case FusedKind::kMatMulGlu:
      return "mul_mat_glu";
    case FusedKind::kMatMulAdd:
      return "mul_mat_add";
    case FusedKind::kOwnerAttention:
      return "owner_attention";
    case FusedKind::kGroupedStores:
      return "grouped_stores";
    case FusedKind::kMoeGlu:
      return "moe_glu";
    case FusedKind::kMoeCombine:
      return "moe_combine";
    case FusedKind::kHcNorm:
      return "hc_norm";
    case FusedKind::kHcNormPost:
      return "hc_norm_post";
    case FusedKind::kQkPrep:
      return "qk_prep";
    case FusedKind::kPreparedProducts:
      return "prepared_products";
    case FusedKind::kNormGate:
      return "norm_gate";
    case FusedKind::kLayerNormModulate:
      return "layer_norm_modulate";
    case FusedKind::kGatedResidual:
      return "gated_residual";
    case FusedKind::kGatedResidualNorm:
      return "gated_residual_norm";
    case FusedKind::kEulerStep:
      return "euler_step";
    case FusedKind::kDupUpAdd:
      return "dup_up_add";
    case FusedKind::kHcPrep:
      return "hc_prep";
    case FusedKind::kScaledUnary:
      return "scaled_unary";
    case FusedKind::kHcFold:
      return "hc_fold";
    case FusedKind::kHcCombine:
      return "hc_combine";
    case FusedKind::kHcMixNorm:
      return "hc_mix_norm";
    case FusedKind::kGdnConv:
      return "gdn_conv";
    case FusedKind::kGdnHistory:
      return "gdn_history";
    case FusedKind::kGdnStep:
      return "gdn_step";
    case FusedKind::kGdnGates:
      return "gdn_gates";
    case FusedKind::kHeadGate:
      return "head_gate";
    case FusedKind::kQsaPool:
      return "qsa_pool";
    case FusedKind::kQsaAttention:
      return "qsa_attention";
    case FusedKind::kMoeRouter:
      return "moe_router";
    case FusedKind::kMoeExperts:
      return "moe_experts";
    case FusedKind::kMatMulIdGlu:
      return "mul_mat_id_glu";
    case FusedKind::kDsv4QHead:
      return "dsv4_qhead";
    case FusedKind::kDsv4Compress:
      return "dsv4_compress";
    case FusedKind::kDsv4HcPre:
      return "dsv4_hc_pre";
    case FusedKind::kStateBankAttention:
      return "state_bank_attention";
  }
  return "unknown";
}

std::string OpKeyName(OpKey key) {
  if (key.op != Op::kFused) {
    return std::string(OpName(key.op));
  }
  return std::string(OpName(key.op)) + "." + std::string(FusedKindName(key.fused));
}

}  // namespace llmp::ir
