// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Every template over the dense decoders: BF16 and F16.

#include "kernels/decode/decoders.cuh"
#include "kernels/decode/launch.h"
#include "kernels/decode/templates.cuh"

namespace llmp::kernels::decode {
namespace {

constexpr DecoderKernels kBf16 = KernelsOf<Bf16Dense>();
constexpr DecoderKernels kF16 = KernelsOf<F16Dense>();

}  // namespace

const DecoderKernels* DenseKernels(DecoderId id) {
  switch (id) {
    case DecoderId::kBf16Dense:
      return &kBf16;
    case DecoderId::kF16Dense:
      return &kF16;
    default:
      return nullptr;
  }
}

}  // namespace llmp::kernels::decode
