// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Every template over NVFP4's decoders: split, CUTLASS's scale atoms,
// row-interleaved and GGML's blocks.

#include "kernels/decode/decoders.cuh"
#include "kernels/decode/launch.h"
#include "kernels/decode/templates.cuh"

namespace llmp::kernels::decode {
namespace {

constexpr DecoderKernels kSplit = KernelsOf<Nvfp4Split>();
constexpr DecoderKernels kAtoms = KernelsOf<Nvfp4Atoms>();
constexpr DecoderKernels kInterleaved = KernelsOf<Nvfp4Interleaved>();
constexpr DecoderKernels kGgmlBlocks = KernelsOf<Nvfp4GgmlBlocks>();

}  // namespace

const DecoderKernels* Nvfp4Kernels(DecoderId id) {
  switch (id) {
    case DecoderId::kNvfp4Split:
      return &kSplit;
    case DecoderId::kNvfp4Atoms:
      return &kAtoms;
    case DecoderId::kNvfp4Interleaved:
      return &kInterleaved;
    case DecoderId::kNvfp4GgmlBlocks:
      return &kGgmlBlocks;
    default:
      return nullptr;
  }
}

}  // namespace llmp::kernels::decode
