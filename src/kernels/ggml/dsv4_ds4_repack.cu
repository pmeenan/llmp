// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <expected>

#include "common.cuh"
#include "kernels/ggml/dsv4_ds4_repack.h"
#include "kernels/ggml/launch.h"

namespace jitllm::kernels::ggml {
namespace {
#include "kernels/ggml/dsv4_ds4_repack_core.cuh"

template <typename T>
T* Pointer(std::uint64_t address) {
  return reinterpret_cast<T*>(static_cast<std::uintptr_t>(address));
}
}  // namespace

std::expected<void, KernelFailure> RunDs4RepackInitialize(LaunchContext& launch,
                                                          const Ds4AlignedShape& shape,
                                                          Ds4CacheBuffer packed) {
  if (auto checked = CheckDs4RepackInitialize(shape, packed); !checked) return checked;
  const auto layout = *Ds4AlignedLayoutOf(shape);
  return launch.Run(base::Bytes(0), [packed, layout](auto& context) {
    CUDA_CHECK(cudaMemsetAsync(Pointer<void>(packed.address), 0, layout.packed_bytes.value(),
                               context.stream()));
  });
}

std::expected<void, KernelFailure> RunDs4RepackChunk(LaunchContext& launch,
                                                     const Ds4RepackChunk& desc) {
  if (auto checked = CheckDs4RepackChunk(desc); !checked) return checked;
  const auto layout = *Ds4AlignedLayoutOf(desc.shape);
  return launch.Run(base::Bytes(0), [desc, layout](auto& context) {
    const auto* raw = Pointer<const unsigned char>(desc.raw.address);
    const auto codes = desc.packed.address + layout.codes_offset;
    switch (desc.shape.kind) {
      case Ds4AlignedKind::kIq2Xxs:
        repack_iq2_xxs_aligned_kernel<<<static_cast<unsigned>((desc.blocks * 8 + 255) / 256), 256,
                                        0, context.stream()>>>(
            Pointer<__half>(desc.packed.address) + desc.first_block,
            Pointer<uint2>(codes) + (desc.first_block * 8), raw, desc.blocks);
        break;
      case Ds4AlignedKind::kQ2K:
        repack_q2_k_aligned_kernel<<<static_cast<unsigned>((desc.blocks * 16 + 255) / 256), 256, 0,
                                     context.stream()>>>(
            Pointer<std::uint32_t>(desc.packed.address),
            Pointer<std::uint32_t>(desc.packed.address + layout.scales_offset),
            Pointer<std::uint32_t>(codes), raw, desc.first_block, desc.blocks,
            layout.blocks_per_row, desc.shape.output);
        break;
      case Ds4AlignedKind::kQ8Dense:
        repack_q8_0_aligned_kernel<<<static_cast<unsigned>((desc.blocks * 2 + 255) / 256), 256, 0,
                                     context.stream()>>>(
            Pointer<__half>(desc.packed.address) + desc.first_block,
            Pointer<unsigned char>(codes) + (desc.first_block * 32), raw, desc.blocks);
        break;
    }
    CUDA_CHECK(cudaGetLastError());
  });
}
}  // namespace jitllm::kernels::ggml
