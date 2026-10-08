// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Approved Q8_0 Gemma4 assistants: semantic weights/target/vocabulary only.
// No execution, assistant KV, process addresses or speculation policy.
#ifndef LLMP_MODEL_GEMMA4_ASSISTANT_H_
#define LLMP_MODEL_GEMMA4_ASSISTANT_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include "model/gemma4.h"

namespace llmp::model {
struct Gemma4AssistantProfile {
  std::string_view name;
  std::uint32_t target_width = 0, heads = 0, local_kv_heads = 0, global_kv_heads = 0;
  std::uint32_t local_target_layer = 0, global_target_layer = 0;
  bool operator==(const Gemma4AssistantProfile&) const = default;
};
const Gemma4AssistantProfile& Gemma4Assistant26();
const Gemma4AssistantProfile& Gemma4Assistant31();
inline constexpr std::uint32_t kGemma4AssistantWidth = 1024;
inline constexpr std::uint32_t kGemma4AssistantFfn = 8192;
inline constexpr std::uint32_t kGemma4AssistantLayers = 4;
inline constexpr std::uint32_t kGemma4AssistantResources = 49;
struct Gemma4AssistantLayer {
  Gemma4Tensor attn_norm, q, q_norm, out, attn_post_norm;
  Gemma4Tensor ffn_norm, gate, up, down, ffn_post_norm, output_scale;
  bool operator==(const Gemma4AssistantLayer&) const = default;
};
struct Gemma4AssistantBinding {
  Gemma4Tensor embedding, head, output_norm, rope_freqs, pre_projection, post_projection;
  std::array<Gemma4AssistantLayer, kGemma4AssistantLayers> layers;
  bool operator==(const Gemma4AssistantBinding&) const = default;
};
std::expected<Gemma4AssistantBinding, std::string> BindGemma4Assistant(
    const Gemma4AssistantProfile& profile, std::string_view architecture,
    std::span<const Gemma4Resource> resources);
std::expected<Gemma4AssistantBinding, std::string> BindGemma4Assistant(
    const Gemma4AssistantProfile& profile, const artifact::Artifact& artifact);
std::expected<void, std::string> CheckGemma4AssistantBinding(const Gemma4AssistantProfile& profile,
                                                             const Gemma4AssistantBinding& binding);
// Checked zero-tensor kept GGUF metadata, at most 16 MiB. Tensor-shape
// binding alone must not substitute for these architecture semantics.
std::expected<void, std::string> CheckGemma4AssistantMetadata(const Gemma4AssistantProfile& profile,
                                                              std::span<const std::byte> metadata);

struct Gemma4AssistantBorrow {
  std::uint32_t target_layer = 0, head_dim = 0, kv_heads = 0, rope_dims = 0;
  bool local = false;
  bool operator==(const Gemma4AssistantBorrow&) const = default;
};
struct Gemma4AssistantTarget {
  std::array<Gemma4AssistantBorrow, 4> layers;
  // Every Q-only block reads the frozen completed target prefix. Draft-chain
  // position stays constant; neither assistant cache writes nor private KV.
  // Input token embedding is target-owned and scaled by sqrt(target width).
  // Input feature is POST target final norm; assistant embedding is its tied
  // full canonical head. Recurrent feature comes from post_projection.
  bool operator==(const Gemma4AssistantTarget&) const = default;
};
// Shape compatibility alone does not prove canonical vocabulary alignment.
// Both mutable bindings are revalidated before producing this semantic map.
std::expected<Gemma4AssistantTarget, std::string> CheckGemma4AssistantTarget(
    const Gemma4AssistantProfile& assistant, const Gemma4AssistantBinding& binding,
    const Gemma4Profile& target, const Gemma4Binding& target_binding);

// Borrowed, caller-funded decoded values: reuse ReadGgufTokenizer for tokens
// and merges; ReadGgufMetadata for raw scores/types (Gemma's TokenizerSpec does
// not retain scores and its token kinds include authority-specific overrides).
// No payload copies/allocations. Admission must run this independently of
// shape binding. The target retains tokenizer/stop authority.
struct Gemma4AssistantVocabulary {
  std::span<const std::string> tokens;
  std::span<const std::pair<std::string, std::string>> merges;
  std::span<const double> scores;
  std::span<const std::int64_t> types;
};
std::expected<void, std::string> CheckGemma4AssistantVocabulary(
    const Gemma4AssistantVocabulary& target, const Gemma4AssistantVocabulary& assistant);
}  // namespace llmp::model
#endif  // LLMP_MODEL_GEMMA4_ASSISTANT_H_
