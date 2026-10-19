// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// DeepSeek V4's architecture mappings (spec/mapping.h), as llama.cpp's
// deepseek4 and dflash GGUFs name their tensors and keys (b29c606e2;
// docs/experiments/dsv4-native/gguf_profile_check.py).
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
// 256-cell steps), every layer routed by its bias;
// its input embedding.inject: its target's token table, and the injection
// from its target's streams' means entering each layer dflash.target_layers
// names (taps features_0 to features_2), projected by fc and normed by
// enc.output_norm; the head's Sinkhorn mix of every row, its target's
// output and the Markov head (markov_w1, markov_w2); conf_proj accounted
// for, unread (llama.cpp's confidence head only shortens drafts below a
// probability floor it leaves at 0). Its import composes it with DeepSeek
// V4 Flash's values and tensors as data here.
//
// Neither mapping folds a weight: llama.cpp's converter keeps DeepSeek V4's
// norms as they are. No default planning policy rule: the shared serving
// route's own recipe (its ServingRecipe's serving_rule) carries
// kDeepSeekServingPolicy, the patterns that reproduce the fast plan.

#ifndef LLMP_SPEC_MAPPINGS_DSV4_H_
#define LLMP_SPEC_MAPPINGS_DSV4_H_

#include <string_view>

#include "spec/mapping.h"

namespace llmp::spec::mappings {

const Mapping& Dsv4Mapping();
const Mapping& DsparkMapping();

// Independent product schedule choices and the caller's owner-row arithmetic
// requirement. Structural eligibility is shared; concrete capabilities and
// authenticated numerical evidence select actual implementations.
inline constexpr std::string_view kDeepSeekServingPolicy =
    "max_decode_rows 16\n"
    "product compact_experts\n"
    "product d2r_experts\n"
    "product dense_pair\n"
    "product pair_experts\n"
    "product pair_glu\n"
    "product pair_glu_q8\n"
    "product vector_floats\n"
    "enable owner_row_invariant\n"
    "when solo past 8 max_decode_rows 8\n"
    "when solo past 8 vector_rows 8\n";

// The block drafter retains ordinary expert pairs and vector float products;
// target-only compact schedules and D2R are independent choices above.
inline constexpr std::string_view kDsparkServingPolicy =
    "max_decode_rows 16\n"
    "product pair_experts\n"
    "product vector_floats\n";

}  // namespace llmp::spec::mappings

#endif  // LLMP_SPEC_MAPPINGS_DSV4_H_
