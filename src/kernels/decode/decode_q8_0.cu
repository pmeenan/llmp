// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Every template over GGML's Q8_0 blocks.

#include "kernels/decode/decoders.cuh"
#include "kernels/decode/launch.h"
#include "kernels/decode/templates.cuh"

namespace llmp::kernels::decode {
namespace {

constexpr DecoderKernels kQ8_0 = KernelsOf<Q8_0Blocks>();

}  // namespace

const DecoderKernels* Q8_0Kernels(DecoderId id) {
  return id == DecoderId::kQ8_0Blocks ? &kQ8_0 : nullptr;
}

}  // namespace llmp::kernels::decode
