// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Every template over the affine integer quantization decoder.

#include "kernels/decode/decoders.cuh"
#include "kernels/decode/launch.h"
#include "kernels/decode/templates.cuh"

namespace llmp::kernels::decode {
namespace {

constexpr DecoderKernels kAffine = KernelsOf<AffineSplit>();

}  // namespace

const DecoderKernels* AffineKernels(DecoderId id) {
  return id == DecoderId::kAffineSplit ? &kAffine : nullptr;
}

}  // namespace llmp::kernels::decode
