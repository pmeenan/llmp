// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Benchmark-only constructor wrapper; absent from production targets.
// JITLLM_BENCH_FULL_FINAL_FFN=0|1 selects the historical narrow/full
// final FFN for exact bounded Gemma4 options. Unset preserves ordinary options.
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <utility>

#include "engine/gemma4_runner.h"

namespace en = jitllm::engine;
extern "C" void
RealGemma4Constructor(en::Gemma4Runner*, en::PagedNode&, en::Gemma4Options, int, std::uint32_t) asm(
    "__real__ZN6jitllm6engine12Gemma4RunnerC1ERNS0_9PagedNodeENS0_13Gemma4OptionsEij");
extern "C" void
WrappedGemma4Constructor(en::Gemma4Runner*, en::PagedNode&, en::Gemma4Options, int, std::uint32_t) asm(
    "__wrap__ZN6jitllm6engine12Gemma4RunnerC1ERNS0_9PagedNodeENS0_13Gemma4OptionsEij");
extern "C" void WrappedGemma4Constructor(en::Gemma4Runner* self, en::PagedNode& node,
                                         en::Gemma4Options options, int owner,
                                         std::uint32_t stream) {
  const auto* supplied = std::getenv("JITLLM_BENCH_FULL_FINAL_FFN");
  const std::string_view mode = supplied == nullptr ? "ordinary" : supplied;
  const bool dense = options.variant == en::Gemma4Variant::k31B;
  if ((supplied != nullptr && mode != "0" && mode != "1") ||
      (!dense && options.variant != en::Gemma4Variant::k26BA4B) || options.context != 8192 ||
      options.max_rows != (dense ? 256U : 1024U) || (options.slots != 1 && options.slots != 4) ||
      options.max_head_rows != options.slots || !options.graphs || options.retain_features ||
      options.reference_masks || options.shared_q8 || !options.fuse_norms ||
      options.row_invariant || options.rope_store || !options.fuse_norm_rope ||
      !options.fuse_norm_add || options.fuse_gemma_route != !dense ||
      options.fuse_gemma_reduce != !dense || options.fuse_quant_glu != dense ||
      !options.owner_attention) {
    std::fputs("frontier control requires exact bounded Gemma4 production options\n", stderr);
    std::exit(2);
  }
  if (supplied != nullptr) options.frontier_head = mode == "0";
  std::fprintf(stderr, "FRONTIER_CONSTRUCTOR_CONTROL full_final_ffn=%d slots=%u head_capacity=%u\n",
               !options.frontier_head, options.slots, options.max_head_rows);
  RealGemma4Constructor(self, node, std::move(options), owner, stream);
}
