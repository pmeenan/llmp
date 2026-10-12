// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Immutable Qwen2 EXL3 source profile and resource validation shared by the
// declarative component mapping and artifact controls. Execution is selected
// per operation by the common engine, not described by a family phase plan.

#ifndef LLMP_MODEL_QWEN2_EXL3_H_
#define LLMP_MODEL_QWEN2_EXL3_H_

#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "model/qwen2.h"

namespace llmp::artifact {
class Artifact;
}

namespace llmp::model {

// Qwen2.5-0.5B-Instruct as the EXL3 fixtures' config.json gives it (the
// same model as Qwen25Instruct05B, with RoPE's original context 32,768 as
// the record's rope parameters have it).
const Qwen2Profile& Qwen25Instruct05BExl3();

// One EXL3 linear's resources, as artifact resource indices, and its shape.
struct Exl3LinearBinding {
  std::string name;  // "model.layers.0.self_attn.q_proj", ..., "lm_head"
  std::uint32_t trellis = 0;
  std::uint32_t suh = 0;
  std::uint32_t svh = 0;
  std::optional<std::uint32_t> bias;  // F16 [n]
  int k = 0;
  int n = 0;
  int bits = 0;
};

struct Exl3LayerBinding {
  std::uint32_t attn_norm = 0;  // BF16 [width]
  std::uint32_t mlp_norm = 0;
  Exl3LinearBinding q, k, v, o, gate, up, down;
};

struct Exl3Binding {
  std::uint32_t embed = 0;       // BF16 [vocab, width]
  std::uint32_t final_norm = 0;  // BF16 [width]
  std::vector<Exl3LayerBinding> layers;
  Exl3LinearBinding lm_head;
};

// A resource as the EXL3 adapter sees it: its name and representation
// (artifact/representation.h): the family ("plain" or "exl3"), the dtype
// and shape, and for EXL3 its role, rate, codebook and features.
struct Exl3Resource {
  std::string name;
  std::string family;
  std::string dtype;
  std::vector<std::uint64_t> shape;
  std::string role;
  std::uint32_t k_bits = 0;
  std::uint64_t in_features = 0;
  std::uint64_t out_features = 0;
  std::string codebook;
};

// Refused, naming the resource, unless the resources are a qwen2 EXL3
// checkpoint's with every tensor the plan reads, of the profile's shapes:
// mcg trellises at K = 4, 5, 6 or 8 with FP16 side vectors and an mcg flag,
// F16 biases on q, k and v only, BF16 norms and embedding; and nothing the
// plan does not read. Resource indices are positions in `resources`.
std::expected<Exl3Binding, std::string> BindQwen2Exl3(const Qwen2Profile& profile,
                                                      std::string_view architecture,
                                                      std::span<const Exl3Resource> resources);
// The same over a validated v0 artifact (artifact/artifact.h).
std::expected<Exl3Binding, std::string> BindQwen2Exl3(const Qwen2Profile& profile,
                                                      const artifact::Artifact& artifact);

}  // namespace llmp::model

#endif  // LLMP_MODEL_QWEN2_EXL3_H_
