// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "base/sha256_internal.h"

#if defined(__aarch64__) && __has_builtin(__builtin_cpu_supports)
#include <arm_neon.h>

#include <bit>

namespace llmp::base::sha256_internal {
namespace {
// Only this function may use SHA2. The translation unit and its dispatcher
// retain the explicit baseline architecture, including on CPUs without SHA2.
__attribute__((target("sha2"))) void Arm(State& state, const std::uint8_t* blocks,
                                         std::size_t count) {
  auto abcd = vld1q_u32(state.data());
  auto efgh = vld1q_u32(state.data() + 4);
  for (std::size_t block = 0; block < count; ++block, blocks += 64) {
    const auto initial_abcd = abcd;
    const auto initial_efgh = efgh;
    uint32x4_t message[4];
    for (std::size_t i = 0; i < 4; ++i) {
      auto bytes = vld1q_u8(blocks + 16 * i);
      if constexpr (std::endian::native == std::endian::little) bytes = vrev32q_u8(bytes);
      message[i] = vreinterpretq_u32_u8(bytes);
    }
    for (std::size_t group = 0; group < 16; ++group) {
      const auto slot = group % 4;
      const auto rounds = vaddq_u32(message[slot], vld1q_u32(kRound.data() + 4 * group));
      const auto previous_abcd = abcd;
      abcd = vsha256hq_u32(abcd, efgh, rounds);
      efgh = vsha256h2q_u32(efgh, previous_abcd, rounds);
      if (group < 12) {
        const auto low = vsha256su0q_u32(message[slot], message[(slot + 1) % 4]);
        message[slot] = vsha256su1q_u32(low, message[(slot + 2) % 4], message[(slot + 3) % 4]);
      }
    }
    abcd = vaddq_u32(abcd, initial_abcd);
    efgh = vaddq_u32(efgh, initial_efgh);
  }
  vst1q_u32(state.data(), abcd);
  vst1q_u32(state.data() + 4, efgh);
}
}  // namespace

Compressor ArmCompressor() { return __builtin_cpu_supports("sha2") ? Arm : nullptr; }
}  // namespace llmp::base::sha256_internal
#else
namespace llmp::base::sha256_internal {
Compressor ArmCompressor() { return nullptr; }
}  // namespace llmp::base::sha256_internal
#endif
