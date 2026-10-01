// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Bit-preserving original ds4 aligned weight preparation. A benchmark
// importer supplies bounded raw chunks and writes prepared files; serving
// maps that physical weight set instead of also retaining the raw experts.
// No GGUF parser, file I/O, allocation or upstream weight server is here.
#ifndef JITLLM_KERNELS_GGML_DSV4_DS4_REPACK_H_
#define JITLLM_KERNELS_GGML_DSV4_DS4_REPACK_H_

#include <cstdint>
#include <expected>

#include "base/bytes.h"
#include "kernels/ggml/dsv4_ds4_cache.h"

namespace jitllm::kernels::ggml {

enum class Ds4AlignedKind : std::uint8_t { kIq2Xxs, kQ2K, kQ8Dense };

struct Ds4AlignedShape {
  Ds4AlignedKind kind = Ds4AlignedKind::kIq2Xxs;
  std::uint32_t input = 0;
  std::uint32_t output = 0;
  std::uint32_t groups = 0;
};

// Same physical sections as original ds4_repack.cu. IQ2/Q8 have half
// scales then a64-byte aligned code section. Q2 has aligned paired d/dmin,
// interleaved scales and interleaved codes. Padding has no numerical role.
struct Ds4AlignedLayout {
  base::Bytes raw_bytes;
  base::Bytes packed_bytes;
  std::uint64_t blocks = 0;
  std::uint64_t scales_offset = 0;  // Q2 scale bytes; otherwise zero
  std::uint64_t codes_offset = 0;
  std::uint32_t blocks_per_row = 0;
  std::uint32_t raw_block_bytes = 0;
};

std::expected<Ds4AlignedLayout, KernelFailure> Ds4AlignedLayoutOf(const Ds4AlignedShape& shape);

struct Ds4RepackChunk {
  Ds4AlignedShape shape{};
  Ds4CacheBuffer raw{};     // dense packed raw blocks beginning at first_block
  Ds4CacheBuffer packed{};  // complete output tensor; one contiguous physical view
  std::uint64_t first_block = 0;
  std::uint64_t blocks = 0;
};

// Q2 chunks follow the original producer's whole row-pair boundary.
// Refuse short/unaligned/overlapping ranges, overflow and unlaunchable grids.
std::expected<void, KernelFailure> CheckDs4RepackChunk(const Ds4RepackChunk& desc);
std::expected<void, KernelFailure> CheckDs4RepackInitialize(const Ds4AlignedShape& shape,
                                                            Ds4CacheBuffer packed);

// CUDA only. Initialize once before any chunks to establish deterministic
// padding bytes. Both calls queue on the caller's jitLLM stream; the caller
// proves completion before writing files, reusing raw staging or releasing
// ownership. Repacking cannot change any source scale or code bit.
std::expected<void, KernelFailure> RunDs4RepackInitialize(LaunchContext& launch,
                                                          const Ds4AlignedShape& shape,
                                                          Ds4CacheBuffer packed);
std::expected<void, KernelFailure> RunDs4RepackChunk(LaunchContext& launch,
                                                     const Ds4RepackChunk& desc);

}  // namespace jitllm::kernels::ggml
#endif  // JITLLM_KERNELS_GGML_DSV4_DS4_REPACK_H_
