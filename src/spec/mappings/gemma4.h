// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Gemma 4's architecture mappings (spec/mapping.h), as llama.cpp's GGUF
// writes the checkpoints (its gemma4.cpp and gemma4-assistant.cpp, read at
// 8019dc56; the retired M3 runners ported b29c606e), and the source values
// of v0 artifacts from the compiled profiles (model/gemma4.h,
// gemma4_assistant.h).
//
// "gemma4" (26B-A4B and 31B): five sliding-window layers to a global one
// (the checkpoint's per-layer flag list must agree); sandwich-normed
// attention with RMS-normed queries and keys, an unweighted RMS norm of V,
// queries unscaled (llama.cpp's f_attention_scale 1) and no attention
// softcap; per-layer-type head widths (the _swa keys for the local
// layers), KV head counts read per layer, rotation dimensions and bases;
// the global layers tie V to K (their checkpoints have no attn_v) and
// rotate proportionally (rope_freqs.weight, a table they share); a layer
// output scale after each layer; a tied output head with a final softcap.
// A checkpoint with experts (gemma4.expert_count above zero) has ffn.moe
// layers: a dense GELU-tanh GLU beside routed experts in a stacked
// gate-up array (a checkpoint with separate gate and up arrays is
// refused), a softmax top-k router over the stream's scaled norm with
// renormalized weights and per-expert down scales; one without has
// ffn.glu layers. The GGUF converter applies no norm fold to Gemma 4
// (llama.cpp's Gemma4Model.norm_shift is 0): its norm weights are used as
// they are, and the mapping asserts no fold.
//
// "gemma4-assistant": the drafter stage "assistant" composed with its
// target (ComposeDrafter): it reads the target's K and V (each local
// layer its target's last local layer's, the global one its last global
// layer's: attention.borrowed, queries only) and the target's final-norm
// output tap "features"; its input is the target's token table's row
// (scaled by sqrt(target width)) beside that feature, projected to its
// width (embedding.projected); then three local and one global layer of
// borrowed attention, GELU-tanh GLU and output scale; its head is its own
// token table (tied, every row an output, no softcap) and projects its
// rows back to the target's width (its next feature).
//
// The default planner policies are the production recipe's pattern
// selection within its qualified envelope (the retired M3 class's
// gemma26/gemma31 production options): the fused norms, norm-rope and
// norm-add on both; the quantized GLU on 31B only; the Gemma route and
// reduce fusions (26B); owner-root attention, common-width reads with
// bounded roots (two one-row owners within 4,096 positions); no
// rope-store, grouped stores or shared Q8 preparation; the assistant's
// plain primitives; and outside the envelope the narrowed final layer.
// A bounded rule also names the prompt chunk the recipe was qualified at
// (spec/mapping.h PolicyRule::prefill_chunk).

#ifndef LLMP_SPEC_MAPPINGS_GEMMA4_H_
#define LLMP_SPEC_MAPPINGS_GEMMA4_H_

#include <expected>
#include <string>
#include <vector>

#include "spec/mapping.h"

namespace llmp::model {
struct Gemma4Profile;
struct Gemma4AssistantProfile;
}  // namespace llmp::model

namespace llmp::spec::mappings {

const Mapping& Gemma4Mapping();
const Mapping& Gemma4AssistantMapping();

// The profile as the GGUF keys the mapping reads (the per-layer lists as
// lists). Refused where the profile says what the mapping cannot (other
// local layers).
std::expected<std::vector<SourceValue>, std::string> Gemma4V0Values(
    const model::Gemma4Profile& profile);

// The assistant's profile, with its target's, as the keys its mapping
// reads. Refused where the profile's target layers are not its target's
// last local and global layers (what the composition gives it).
std::expected<std::vector<SourceValue>, std::string> Gemma4AssistantV0Values(
    const model::Gemma4AssistantProfile& assistant, const model::Gemma4Profile& target);

}  // namespace llmp::spec::mappings

#endif  // LLMP_SPEC_MAPPINGS_GEMMA4_H_
