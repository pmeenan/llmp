// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Every template over MXFP8's decoders: split and CUTLASS's scale atoms.

#include "kernels/decode/decoders.cuh"
#include "kernels/decode/launch.h"
#include "kernels/decode/templates.cuh"

namespace llmp::kernels::decode {
namespace {

constexpr DecoderKernels kSplit = KernelsOf<Mxfp8Split>();
constexpr DecoderKernels kAtoms = KernelsOf<Mxfp8Atoms>();

}  // namespace

const DecoderKernels* Mxfp8Kernels(DecoderId id) {
  switch (id) {
    case DecoderId::kMxfp8Split:
      return &kSplit;
    case DecoderId::kMxfp8Atoms:
      return &kAtoms;
    default:
      return nullptr;
  }
}

}  // namespace llmp::kernels::decode
