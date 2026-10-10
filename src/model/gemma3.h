// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Approved Gemma 3 4B QAT profile and checked tensor binding: the
// approval a prepared artifact passes before admission. The state layout,
// step inputs and graphs come from the components (D-107, spec/mappings/
// gemma3.h). Media inputs remain outside this text profile.
#ifndef LLMP_MODEL_GEMMA3_H_
#define LLMP_MODEL_GEMMA3_H_

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace llmp::artifact {
class Artifact;
}
namespace llmp::model {

struct Gemma3Profile {
  std::uint32_t layers = 0, width = 0, ffn = 0, heads = 0, kv_heads = 0;
  std::uint32_t key_dim = 0, value_dim = 0, vocab = 0, context = 0, window = 0;
  float rope_base = 0, rope_scale = 0, rms_eps = 0;
  std::string_view rope_scaling;
  bool operator==(const Gemma3Profile&) const = default;
  // d812 gemma3.cpp load_swa_pattern(6): the final layer33 is local.
  bool local(std::uint32_t layer) const { return (layer + 1) % 6 != 0; }
};
const Gemma3Profile& Gemma3_4BQat();

struct Gemma3Resource {
  std::vector<std::string> roles;
  std::string type;
  std::vector<std::uint64_t> ne;
  std::uint64_t readable = 0;
};
struct Gemma3Tensor {
  std::uint32_t index = 0;
  std::string type;
  std::vector<std::uint64_t> ne;
  std::uint64_t readable = 0;
  bool operator==(const Gemma3Tensor&) const = default;
};
struct Gemma3Layer {
  Gemma3Tensor attn_norm, q, k, v, out, q_norm, k_norm, attn_post_norm;
  Gemma3Tensor ffn_norm, gate, up, down, ffn_post_norm;
  bool operator==(const Gemma3Layer&) const = default;
};
struct Gemma3Binding {
  // The approved file has no output.weight; output aliases token_embd.
  Gemma3Tensor token_embd, output, output_norm;
  std::vector<Gemma3Layer> layers;
  bool operator==(const Gemma3Binding&) const = default;
};
// Closed to the approved profile and actual F32/Q4_0/Q8_0 tensor contract.
// This recognizes storage, not executable kernel or importer support.
std::expected<Gemma3Binding, std::string> BindGemma3(const Gemma3Profile& profile,
                                                     std::string_view architecture,
                                                     std::span<const Gemma3Resource> resources);
std::expected<Gemma3Binding, std::string> BindGemma3(const Gemma3Profile& profile,
                                                     const artifact::Artifact& artifact);
// Execution admission additionally authenticates the approved prepared source.
std::expected<Gemma3Binding, std::string> BindApprovedGemma3(const artifact::Artifact& artifact);
// Public descriptors are rechecked before any graph or placement.
std::expected<void, std::string> CheckGemma3Binding(const Gemma3Profile& profile,
                                                    const Gemma3Binding& binding);

}  // namespace llmp::model
#endif  // LLMP_MODEL_GEMMA3_H_
