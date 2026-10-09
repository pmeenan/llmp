// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The GLU over pairs of different decoders (D-108: fused kernels take a
// decoder per input, so differently encoded operands fuse): MXFP8 gate
// with a BF16 up, and an NVFP4 gate with a Q8_0 up. Further pairs are an
// instantiation each, added where a model mixes them.

#include "kernels/decode/decoders.cuh"
#include "kernels/decode/launch.h"
#include "kernels/decode/templates.cuh"

namespace llmp::kernels::decode {

Launcher MixedGluLauncher(DecoderId gate, DecoderId up) {
  if (gate == DecoderId::kMxfp8Split && up == DecoderId::kBf16Dense) {
    return &LaunchVectorGlu<Mxfp8Split, Bf16Dense>;
  }
  if (gate == DecoderId::kNvfp4Split && up == DecoderId::kQ8_0Blocks) {
    return &LaunchVectorGlu<Nvfp4Split, Q8_0Blocks>;
  }
  return nullptr;
}

}  // namespace llmp::kernels::decode
