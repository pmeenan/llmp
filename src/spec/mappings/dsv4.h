// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// DeepSeek V4's architecture mappings (spec/mapping.h), as llama.cpp's
// deepseek4 and dflash GGUFs name their tensors and keys (b29c606e2;
// docs/experiments/dsv4-native/gguf_profile_check.py), and the source
// values of v0 artifacts from today's compiled profiles (model/dsv4.h,
// model/dspark.h).
//
// "deepseek4" (DeepSeek V4 Flash): hyper-connection streams mixed by
// Sinkhorn (residual.hc; each block between hc.mix and hc.combine, the
// mix's input norm the block's own attn_norm or ffn_norm); each layer's
// attention.mqa by its compression ratio (deepseek4.attention.
// compress_ratios gives each layer's type: 0 'w' the window alone at the
// plain rotation base; 4 'c' CSA, overlapping compressor and lightning
// indexer; 128 'h' HCA; the compressed layers at their compressed base
// with YaRN), its window cache a ring; every layer's MoE (ffn.moe,
// placement none: sqrt(softplus) routing with a learned bias, renormalized
// and scaled weights, clamped SwiGLU experts and the ungated shared
// expert), its first deepseek4.hash_layer_count layers routed by a
// token-hash side input each (side.hash "hash_<layer>", the layer's
// ffn_gate_tid2eid); the head's Sinkhorn mix with the final norm and the
// untied output head.
//
// "dflash" (the DSpark drafter, stage "dspark"): composed with its target
// (ComposeDrafter); three window-only DeepSeek V4 blocks over an injected
// ring of 256 cells (the window and a draft block, padded to attention's
// 256-cell steps: model/dspark.h's ring), every layer routed by its bias;
// its input embedding.inject: its target's token table, and the injection
// from its target's streams' means entering each layer dflash.target_layers
// names (taps features_0 to features_2), projected by fc and normed by
// enc.output_norm; the head's Sinkhorn mix of every row, its target's
// output and the Markov head (markov_w1, markov_w2); conf_proj accounted
// for, unread (llama.cpp's confidence head only shortens drafts below a
// probability floor it leaves at 0).
//
// Neither mapping folds a weight: llama.cpp's converter keeps DeepSeek V4's
// norms as they are. No default planning policy yet: the patterns that
// reproduce today's fast plan (the hyper-connection kernels, the Q head,
// the compressor, the indexer's selection, the sparse attention, output-A,
// the routing and the fused experts) are slice G4b's.

#ifndef LLMP_SPEC_MAPPINGS_DSV4_H_
#define LLMP_SPEC_MAPPINGS_DSV4_H_

#include <expected>
#include <string>
#include <vector>

#include "spec/mapping.h"

namespace llmp::model {
struct Dsv4Profile;
struct DsparkProfile;
}  // namespace llmp::model

namespace llmp::spec::mappings {

const Mapping& Dsv4Mapping();
const Mapping& DsparkMapping();

// The profile as the GGUF keys the mapping reads. Refused where the
// profile says what the mapping cannot (another layer form).
std::expected<std::vector<SourceValue>, std::string> Dsv4V0Values(
    const model::Dsv4Profile& profile);

// The DSpark drafter's, as its mapping's keys.
std::expected<std::vector<SourceValue>, std::string> DsparkV0Values(
    const model::DsparkProfile& profile);

// The planning policy for DeepSeek serving fast forms.
inline constexpr std::string_view kDeepSeekServingPolicy =
    "max_decode_rows 16\n"
    "enable dsv4_compress\n"
    "enable dsv4_hc_post\n"
    "enable dsv4_hc_pre\n"
    "enable dsv4_qhead\n"
    "enable owner_row_invariant\n"
    "enable state_bank_attention\n"
    "enable vecq_glu\n"
    "enable vecq_inputs\n"
    "disable grouped_stores\n"
    "disable mul_mat_glu\n"
    "disable narrow_final\n"
    "disable owner_attention\n"
    "disable prepared_inputs\n"
    "disable rms_norm_mul_add\n"
    "disable rms_norm_mul_rope\n"
    "disable rope_store\n"
    "disable route_reduce\n";

}  // namespace llmp::spec::mappings

#endif  // LLMP_SPEC_MAPPINGS_DSV4_H_
