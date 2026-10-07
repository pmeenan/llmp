// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Approved Gemma 2 2B Q8_0 profile and checked binding.
// No media adapter, runner or serving support.
#ifndef JITLLM_MODEL_GEMMA2_H_
#define JITLLM_MODEL_GEMMA2_H_

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace jitllm::artifact {
class Artifact;
}
namespace jitllm::model {

struct Gemma2Profile {
  std::uint32_t layers = 0, width = 0, ffn = 0, heads = 0, kv_heads = 0;
  std::uint32_t key_dim = 0, value_dim = 0, vocab = 0, context = 0, window = 0;
  float rope_base = 0, rope_scale = 0, rms_eps = 0;
  float attention_scale = 0, attention_softcap = 0, final_softcap = 0;
  bool operator==(const Gemma2Profile&) const = default;
  // d812 gemma2.cpp load_swa_pattern(2): even layers local, odd global.
  bool local(std::uint32_t layer) const { return layer % 2 == 0; }
};
const Gemma2Profile& Gemma2_2B();

struct Gemma2Resource {
  std::vector<std::string> roles;
  std::string type;
  std::vector<std::uint64_t> ne;
  std::uint64_t readable = 0;
};
struct Gemma2Tensor {
  std::uint32_t index = 0;
  std::string type;
  std::vector<std::uint64_t> ne;
  std::uint64_t readable = 0;
  bool operator==(const Gemma2Tensor&) const = default;
};
struct Gemma2Layer {
  Gemma2Tensor attn_norm, q, k, v, out, attn_post_norm;
  Gemma2Tensor ffn_norm, gate, up, down, ffn_post_norm;
  bool operator==(const Gemma2Layer&) const = default;
};
struct Gemma2Binding {
  // The approved file has no output.weight; output aliases token_embd.
  Gemma2Tensor token_embd, output, output_norm;
  std::vector<Gemma2Layer> layers;
  bool operator==(const Gemma2Binding&) const = default;
};
// Closed to the approved profile and actual F32/Q8_0 tensor contract.
// This recognizes storage, not executable kernel or importer support.
std::expected<Gemma2Binding, std::string> BindGemma2(const Gemma2Profile& profile,
                                                     std::string_view architecture,
                                                     std::span<const Gemma2Resource> resources);
std::expected<Gemma2Binding, std::string> BindGemma2(const Gemma2Profile& profile,
                                                     const artifact::Artifact& artifact);
// Public descriptors are rechecked before any graph or placement.
std::expected<void, std::string> CheckGemma2Binding(const Gemma2Profile& profile,
                                                    const Gemma2Binding& binding);

}  // namespace jitllm::model
#endif  // JITLLM_MODEL_GEMMA2_H_
