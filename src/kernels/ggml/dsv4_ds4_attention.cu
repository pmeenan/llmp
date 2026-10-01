// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cub/block/block_radix_sort.cuh>
#include <expected>
#include <limits>
#include <string>
#include <utility>

#include "base/bytes.h"
#include "common.cuh"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/dsv4_ds4_attention.h"
#include "kernels/ggml/ggml_support.h"
#include "kernels/ggml/launch.h"

namespace jitllm::kernels::ggml {
namespace {

#include "kernels/ggml/dsv4_ds4_attention_core.cuh"

static_assert(sizeof(ds4_decode_scalars) == sizeof(Ds4AttentionDecodeScalars));
static_assert(sizeof(ds4_layer_scalars) == sizeof(Ds4IndexerLayerScalars));
constexpr std::uint64_t kTokentileSmem = tt_TokentileSmemBudget<kTTStageRows, kTTG>::total;

template <typename T>
T* Pointer(const Ds4CacheBuffer& buffer) {
  return reinterpret_cast<T*>(static_cast<std::uintptr_t>(buffer.address));
}

template <typename T>
T* Scratch(const Ds4Attention& desc, std::uint64_t offset) {
  return reinterpret_cast<T*>(Pointer<unsigned char>(desc.scratch) + offset);
}

bool Absent(const Ds4CacheBuffer& buffer) { return buffer.address == 0 && buffer.bytes == 0; }
bool Indexed(const Ds4Attention& desc) { return desc.domain == Ds4AttentionDomain::kIndexedRing; }
bool Static(const Ds4Attention& desc) {
  return desc.domain == Ds4AttentionDomain::kRawPrefill ||
         desc.domain == Ds4AttentionDomain::kMixedPrefill ||
         desc.domain == Ds4AttentionDomain::kMaskedPrefill;
}

std::unexpected<KernelFailure> Rejected(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}

struct Choice {
  Ds4AttentionKind kind;
  Ds4AttentionScratch scratch;
};

std::uint32_t CompressedCeiling(const Ds4Attention& desc) {
  return Absent(desc.layer_scalars) ? desc.compressed_count : desc.compressed_cells;
}

bool Fits(const Ds4Attention& desc, const Ds4AttentionScratch& plan) {
  return plan.bytes == 0 || (!Absent(desc.scratch) && desc.scratch.bytes >= plan.bytes);
}

std::expected<Choice, KernelFailure> Choose(const LaunchContext& launch, const Ds4Attention& desc) {
  const auto& device = ggml_cuda_info().devices[launch.device()];
  const auto sms = static_cast<std::uint32_t>(device.nsm);
  const auto resolve = [&](Ds4AttentionKind kind,
                           bool predecode = true) -> std::expected<Choice, KernelFailure> {
    auto plan = PlanDs4AttentionScratch(desc, kind, sms, predecode);
    if (!plan) return std::unexpected(plan.error());
    if (!Fits(desc, *plan)) {
      // Original predecode scratch OOM retains in-kernel packed reads.
      // Required ID sorting still must fit before anything is submitted.
      if (kind == Ds4AttentionKind::kScalar && plan->predecode) {
        (void)(plan = PlanDs4AttentionScratch(desc, kind, sms, false));
        if (!plan) return std::unexpected(plan.error());
      }
      if (!Fits(desc, *plan)) return Rejected("ds4 complete attention chain lacks paid scratch");
    }
    if (kind == Ds4AttentionKind::kTokenTile &&
        (launch.capturing() || device.cc < 800 || device.smpbo < kTokentileSmem))
      return Rejected("original ds4 token tile needs eager sm_80 and its shared-memory budget");
    if (kind == Ds4AttentionKind::kCublas && !launch.cublas())
      return Rejected("original ds4 SGEMM requires a native lent handle");
    if (kind == Ds4AttentionKind::kScalar && !Indexed(desc) && !Static(desc) &&
        CompressedCeiling(desc) > 7936)
      return Rejected("original ds4 fixed-score kernel cannot serve the live compressed band");
    return Choice{kind, *plan};
  };
  if (desc.kind != Ds4AttentionKind::kOriginalDefault) return resolve(desc.kind);

  // The indexed HG gather precedes all predecode/sort/online paths.
  if (Indexed(desc) && desc.tokens <= 8 && !Absent(desc.positions)) {
    if (auto choice = resolve(Ds4AttentionKind::kHeadGroup); choice) return choice;
  }
  // Each original token-tile dispatch has identical borrowed producer costs.
  if (auto choice = resolve(Ds4AttentionKind::kTokenTile); choice) return choice;

  const bool per_row = !Absent(desc.positions) || !Absent(desc.bank_ids);
  const bool mseq_heads8 = per_row && desc.allow_multisequence_heads8 && desc.tokens >= 128 &&
                           Absent(desc.draft_raw_count) && Absent(desc.decode_scalars) &&
                           Absent(desc.layer_scalars);
  const bool force_main =
      (per_row && !mseq_heads8) || !Absent(desc.decode_scalars) || !Absent(desc.layer_scalars);
  if (Indexed(desc)) {
    if (!force_main && desc.tokens > 1) return resolve(Ds4AttentionKind::kHeads8Online);
    return resolve(Ds4AttentionKind::kScalar);
  }
  if (Static(desc)) {
    if (Absent(desc.mask) && desc.tokens >= 128 && !desc.quality_mode)
      return resolve(Ds4AttentionKind::kHeads8Online);
    if (desc.tokens > 1 && launch.cublas()) return resolve(Ds4AttentionKind::kCublas);
    return resolve(Ds4AttentionKind::kScalar);
  }
  const bool hg = desc.tokens <= 8 && Absent(desc.mask) && Absent(desc.draft_raw_count);
  if (CompressedCeiling(desc) > 7936) {
    if (hg) {
      if (auto choice = resolve(Ds4AttentionKind::kHeadGroup); choice) return choice;
    }
    if (Absent(desc.mask) && Absent(desc.draft_raw_count))
      return resolve(Ds4AttentionKind::kHeads8Online);
    return Rejected("original ds4 deep attention has no mask/draft-span port");
  }
  if (!force_main && Absent(desc.mask) && Absent(desc.draft_raw_count) && desc.tokens > 1 &&
      (mseq_heads8 || (desc.tokens >= 128 && !desc.quality_mode)))
    return resolve(Ds4AttentionKind::kHeads8Online);
  if (hg) {
    if (auto choice = resolve(Ds4AttentionKind::kHeadGroup); choice) return choice;
  }
  if (desc.domain == Ds4AttentionDomain::kDecodeHeads && device.nsm > 64) {
    if (auto choice = resolve(Ds4AttentionKind::kPerHeadSplit); choice) return choice;
  }
  return resolve(Ds4AttentionKind::kScalar);
}

void QueueTokenTile(ggml_backend_cuda_context& context, const Ds4Attention& desc,
                    const Ds4AttentionScratch& plan) {
  auto* records = Scratch<int2>(desc, plan.records);
  auto* counts = Scratch<std::uint32_t>(desc, plan.counts);
  auto* raw_mirror = Scratch<half>(desc, plan.raw_mirror);
  auto* comp_mirror = Scratch<half>(desc, plan.compressed_mirror);
  const auto* positions = Pointer<const std::int32_t>(desc.positions);
  const auto* banks = Pointer<const std::int32_t>(desc.bank_ids);
  const auto first = Static(desc) ? 0 : desc.consecutive_first;
  const auto tiles = (desc.tokens + 3) / 4;
  const auto mirror_rows = desc.tokens + 127;
  std::uint32_t available_before = 0;
  std::uint32_t first_raw_position = 0;
  if (positions) {
    available_before = std::min(first, std::uint32_t{127});
  } else {
    available_before = std::min(desc.raw_count - desc.tokens, desc.first);
    first_raw_position = static_cast<std::uint32_t>(static_cast<std::uint64_t>(desc.first) +
                                                    desc.tokens - desc.raw_count);
  }
  const auto raw_min = 127 - std::min(available_before, std::uint32_t{127});
  if (Indexed(desc)) {
    if (desc.compressed_count <= 32768) {
      const auto shared = static_cast<std::size_t>((desc.compressed_count + 1) >> 1) * 4;
      attention_tokentile_union_build_kernel<<<tiles, 512, shared, context.stream()>>>(
          records, counts, Pointer<const std::int32_t>(desc.selected), positions, desc.first,
          desc.tokens, desc.top_k, desc.ratio, desc.compressed_count, plan.record_stride);
    } else {
      attention_tokentile_union_sort_kernel<<<tiles, 512, 0, context.stream()>>>(
          records, counts, Pointer<const std::int32_t>(desc.selected), positions, desc.first,
          desc.tokens, desc.top_k, desc.ratio, desc.compressed_count, plan.record_stride);
    }
  } else {
    attention_tokentile_dense_build_kernel<<<tiles, 512, 0, context.stream()>>>(
        records, counts, positions, desc.first, desc.tokens, desc.ratio, desc.compressed_count,
        plan.record_stride);
  }
  CUDA_CHECK(cudaGetLastError());
  attention_tokentile_raw_mirror_kernel<<<mirror_rows, 256, 0, context.stream()>>>(
      raw_mirror, Pointer<const float>(desc.raw), banks, first, desc.tokens, desc.raw_cells,
      desc.raw_start, first_raw_position, raw_min, 512);
  CUDA_CHECK(cudaGetLastError());
  if (desc.compressed_count != 0) {
    attention_tokentile_comp_mirror_kernel<<<desc.compressed_count, 256, 0, context.stream()>>>(
        comp_mirror, Pointer<const float>(desc.compressed),
        Pointer<const unsigned char>(desc.compressed_codes),
        Pointer<const float>(desc.compressed_scales), banks, desc.compressed_cells,
        desc.compressed_count, 512, Pointer<const float>(desc.decode_table),
        Pointer<Ds4AttentionDiagnostics>(desc.diagnostics));
    CUDA_CHECK(cudaGetLastError());
  }
  const dim3 grid(tiles, 8, 1);
  attention_tokentile_hmma_kernel<<<grid, 512, kTokentileSmem, context.stream()>>>(
      Pointer<float>(desc.output), Pointer<const float>(desc.sinks),
      Pointer<const float>(desc.query), raw_mirror, comp_mirror, records, counts,
      plan.record_stride, desc.tokens, 64, raw_min);
  CUDA_CHECK(cudaGetLastError());
}

void QueueSplit(ggml_backend_cuda_context& context, const Ds4Attention& desc,
                const Choice& choice) {
  const auto* query = Pointer<const float>(desc.query);
  const auto* raw = Pointer<const float>(desc.raw);
  const auto* comp = Pointer<const float>(desc.compressed_cells != 0 ? desc.compressed : desc.raw);
  const auto* positions = Pointer<const std::int32_t>(desc.positions);
  const auto* banks = Pointer<const std::int32_t>(desc.bank_ids);
  const auto* live = Pointer<const ds4_decode_scalars>(desc.decode_scalars);
  const auto* layer = Pointer<const ds4_layer_scalars>(desc.layer_scalars);
  const auto* codes = Pointer<const unsigned char>(desc.compressed_codes);
  const auto* scales = Pointer<const float>(desc.compressed_scales);
  const auto* table = Pointer<const float>(desc.decode_table);
  auto* diagnostics = Pointer<Ds4AttentionDiagnostics>(desc.diagnostics);
  auto* partials = Scratch<float>(desc, choice.scratch.partials);
  const auto splits = choice.scratch.splits;
  if (choice.kind == Ds4AttentionKind::kPerHeadSplit) {
    const dim3 grid(splits, 64, 1);
    attention_decode_split_kernel<<<grid, 256, 0, context.stream()>>>(
        partials, query, raw, comp, Pointer<const float>(desc.mask),
        static_cast<std::uint32_t>(!Absent(desc.mask)), desc.raw_count, desc.raw_cells,
        desc.raw_start, desc.compressed_count, 64, 512, splits, live, layer, codes, scales, table,
        diagnostics);
    CUDA_CHECK(cudaGetLastError());
    attention_decode_combine_kernel<<<64, 256, 0, context.stream()>>>(
        Pointer<float>(desc.output), Pointer<const float>(desc.sinks), partials, 64, 512, splits);
    CUDA_CHECK(cudaGetLastError());
    return;
  }
  const dim3 grid(splits, 8, desc.tokens);
  if (Indexed(desc)) {
    attention_indexed_hg_partial_kernel<<<grid, 256, 0, context.stream()>>>(
        partials, query, raw, comp, Pointer<const std::int32_t>(desc.selected), desc.tokens,
        desc.raw_cells, desc.compressed_count, desc.compressed_cells, desc.top_k, desc.window,
        desc.ratio, 64, 512, splits, positions, banks, live, layer, codes, scales, table,
        diagnostics);
  } else {
    attention_decode_hg_partial_kernel<<<grid, 256, 0, context.stream()>>>(
        partials, query, raw, comp, desc.tokens, desc.first, desc.raw_count, desc.raw_cells,
        desc.raw_start, desc.compressed_count,
        desc.domain == Ds4AttentionDomain::kDecodeHeads ? 0 : desc.compressed_cells, desc.window,
        desc.ratio, 64, 512, splits, positions, banks, live, layer, codes, scales, table,
        diagnostics);
  }
  CUDA_CHECK(cudaGetLastError());
  const dim3 combine(desc.tokens, 64, 1);
  attention_decode_hg_combine_kernel<<<combine, 256, 0, context.stream()>>>(
      Pointer<float>(desc.output), Pointer<const float>(desc.sinks), partials, 64, 512, splits);
  CUDA_CHECK(cudaGetLastError());
}

void QueueIndexed(ggml_backend_cuda_context& context, const Ds4Attention& desc,
                  const Choice& choice) {
  const auto* query = Pointer<const float>(desc.query);
  const auto* raw = Pointer<const float>(desc.raw);
  const auto* comp = Pointer<const float>(desc.compressed);
  const auto* positions = Pointer<const std::int32_t>(desc.positions);
  const auto* banks = Pointer<const std::int32_t>(desc.bank_ids);
  const auto* live = Pointer<const ds4_decode_scalars>(desc.decode_scalars);
  const auto* layer = Pointer<const ds4_layer_scalars>(desc.layer_scalars);
  const auto* codes = Pointer<const unsigned char>(desc.compressed_codes);
  const auto* scales = Pointer<const float>(desc.compressed_scales);
  const auto* table = Pointer<const float>(desc.decode_table);
  auto* diagnostics = Pointer<Ds4AttentionDiagnostics>(desc.diagnostics);
  const auto* selected = Pointer<const std::int32_t>(desc.selected);
  const auto& plan = choice.scratch;
  if (plan.sort_ids) {
    auto* sorted = Scratch<std::int32_t>(desc, plan.sorted_ids);
    indexed_topk_sort_512_asc_kernel<<<desc.tokens, 512, 0, context.stream()>>>(sorted, selected,
                                                                                desc.tokens);
    CUDA_CHECK(cudaGetLastError());
    selected = sorted;
  }
  if (choice.kind == Ds4AttentionKind::kHeads8Online) {
    const dim3 grid(desc.tokens, 4, 1);
    attention_indexed_mixed_heads8_online_kernel<8, 16><<<grid, 512, 0, context.stream()>>>(
        Pointer<float>(desc.output), Pointer<const float>(desc.sinks), query, raw, comp, selected,
        desc.tokens, desc.first, desc.raw_count, desc.raw_cells, desc.raw_start,
        desc.compressed_count, desc.top_k, desc.window, desc.ratio, 64, 512, positions, banks,
        desc.compressed_cells, codes, scales, table, diagnostics);
    CUDA_CHECK(cudaGetLastError());
    return;
  }
  if (choice.kind == Ds4AttentionKind::kIndexedTwoPass) {
    const dim3 grid(desc.tokens, 8, 1);
    attention_indexed_mixed_heads8_rb4_kernel<<<grid, 256, 0, context.stream()>>>(
        Pointer<float>(desc.output), Pointer<const float>(desc.sinks), query, raw, comp, selected,
        desc.tokens, desc.first, desc.raw_count, desc.raw_cells, desc.raw_start,
        desc.compressed_count, desc.top_k, desc.window, desc.ratio, 64, 512, codes, scales, table,
        diagnostics);
    CUDA_CHECK(cudaGetLastError());
    return;
  }
  const float* slot_predecode = nullptr;
  if (plan.predecode) {
    auto* decoded = Scratch<float>(desc, plan.predecoded);
    const dim3 predecode(desc.tokens, desc.top_k, 1);
    if (!positions) {
      attention_fp8_predecode_kernel<<<predecode, 128, 0, context.stream()>>>(
          decoded, selected, codes, scales, desc.tokens, desc.first, desc.compressed_count,
          desc.top_k, 512, desc.ratio, live, layer, table, diagnostics);
      comp = decoded;
    } else {
      attention_fp8_predecode_tslot_kernel<<<predecode, 128, 0, context.stream()>>>(
          decoded, selected, codes, scales, desc.tokens, desc.compressed_count, desc.top_k, 512,
          desc.ratio, positions, banks, desc.compressed_cells, table, diagnostics);
      slot_predecode = decoded;
    }
    CUDA_CHECK(cudaGetLastError());
    codes = nullptr;
    scales = nullptr;
  }
  const dim3 grid(desc.tokens, 64, 1);
  attention_indexed_mixed_kernel<<<grid, 256, 0, context.stream()>>>(
      Pointer<float>(desc.output), Pointer<const float>(desc.sinks), query, raw, comp, selected,
      desc.tokens, desc.first, desc.raw_count, desc.raw_cells, desc.raw_start,
      desc.compressed_count, desc.compressed_cells, desc.top_k, desc.window, desc.ratio, 64, 512,
      positions, banks, live, layer, codes, scales, slot_predecode, table, diagnostics);
  CUDA_CHECK(cudaGetLastError());
}

void QueueCublas(ggml_backend_cuda_context& context, const Ds4Attention& desc,
                 const Ds4AttentionScratch& plan) {
  auto* handle = internal::CublasHandleOf(context);
  const bool raw_only = desc.domain == Ds4AttentionDomain::kRawPrefill;
  const auto key_count = desc.tokens + (raw_only ? 0 : desc.compressed_count);
  const float* keys = Pointer<const float>(desc.raw);
  if (!raw_only) {
    auto* packed = Scratch<float>(desc, plan.gemm_keys);
    const auto elements = static_cast<std::uint64_t>(key_count) * 512;
    attention_prefill_pack_mixed_kv_kernel<<<static_cast<unsigned int>((elements + 255) / 256), 256,
                                             0, context.stream()>>>(
        packed, Pointer<const float>(desc.raw),
        Pointer<const float>(desc.compressed_count != 0 ? desc.compressed : desc.raw), desc.tokens,
        desc.compressed_count, 512, Pointer<const unsigned char>(desc.compressed_codes),
        Pointer<const float>(desc.compressed_scales), Pointer<const float>(desc.decode_table),
        Pointer<Ds4AttentionDiagnostics>(desc.diagnostics));
    CUDA_CHECK(cudaGetLastError());
    keys = packed;
  }
  auto* scores = Scratch<float>(desc, plan.gemm_scores);
  auto* result = Scratch<float>(desc, plan.gemm_output);
  const float alpha = rsqrtf(512.0f);
  const float zero = 0.0f;
  CUBLAS_CHECK(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(key_count), static_cast<int>(desc.tokens),
      512, &alpha, keys, 512, 0, Pointer<const float>(desc.query), 64 * 512, 512, &zero, scores,
      static_cast<int>(key_count), static_cast<long long>(key_count) * desc.tokens, 64));
  const dim3 softmax(desc.tokens, 64, 1);
  if (raw_only) {
    attention_prefill_raw_softmax_kernel<<<softmax, 256, 0, context.stream()>>>(
        scores, Pointer<const float>(desc.sinks), desc.tokens, desc.window, key_count);
  } else {
    attention_prefill_mixed_softmax_kernel<<<softmax, 256, 0, context.stream()>>>(
        scores, Pointer<const float>(desc.sinks), Pointer<const float>(desc.mask),
        static_cast<std::uint32_t>(!Absent(desc.mask)), desc.tokens, desc.compressed_count,
        desc.window, desc.ratio, key_count);
  }
  CUDA_CHECK(cudaGetLastError());
  const float one = 1.0f;
  CUBLAS_CHECK(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, 512, static_cast<int>(desc.tokens),
      static_cast<int>(key_count), &one, keys, 512, 0, scores, static_cast<int>(key_count),
      static_cast<long long>(key_count) * desc.tokens, &zero, result, 512,
      static_cast<long long>(512) * desc.tokens, 64));
  const auto elements = static_cast<std::uint64_t>(desc.tokens) * 64 * 512;
  attention_prefill_unpack_heads_kernel<<<static_cast<unsigned int>((elements + 255) / 256), 256, 0,
                                          context.stream()>>>(Pointer<float>(desc.output), result,
                                                              desc.tokens, 64, 512);
  CUDA_CHECK(cudaGetLastError());
}

void QueueOther(ggml_backend_cuda_context& context, const Ds4Attention& desc,
                Ds4AttentionKind kind) {
  auto* output = Pointer<float>(desc.output);
  const auto* query = Pointer<const float>(desc.query);
  const auto* sinks = Pointer<const float>(desc.sinks);
  const auto* raw = Pointer<const float>(desc.raw);
  const auto* comp = Pointer<const float>(desc.compressed_cells != 0 ? desc.compressed : desc.raw);
  const auto* codes = Pointer<const unsigned char>(desc.compressed_codes);
  const auto* scales = Pointer<const float>(desc.compressed_scales);
  const auto* table = Pointer<const float>(desc.decode_table);
  auto* diagnostics = Pointer<Ds4AttentionDiagnostics>(desc.diagnostics);
  if (Static(desc)) {
    if (kind == Ds4AttentionKind::kHeads8Online) {
      const dim3 grid(desc.tokens, 8, 1);
      attention_static_mixed_heads8_online_kernel<<<grid, 256, 0, context.stream()>>>(
          output, sinks, query, raw, comp, desc.tokens, desc.compressed_count, desc.window,
          desc.domain == Ds4AttentionDomain::kRawPrefill ? 1 : desc.ratio, 64, 512, codes, scales,
          table, diagnostics);
    } else {
      const dim3 grid(desc.tokens, 64, 1);
      if (desc.domain == Ds4AttentionDomain::kRawPrefill) {
        attention_prefill_raw_kernel<<<grid, 128, 0, context.stream()>>>(
            output, sinks, query, raw, desc.tokens, desc.window, 64, 512);
      } else {
        attention_prefill_mixed_kernel<<<grid, 256, 0, context.stream()>>>(
            output, sinks, query, raw, comp, Pointer<const float>(desc.mask),
            static_cast<std::uint32_t>(!Absent(desc.mask)), desc.tokens, desc.compressed_count,
            desc.window, desc.ratio, 64, 512, codes, scales, table, diagnostics);
      }
    }
  } else if (kind == Ds4AttentionKind::kHeads8Online) {
    const dim3 grid(desc.tokens, 8, 1);
    attention_decode_mixed_heads8_online_kernel<<<grid, 256, 0, context.stream()>>>(
        output, sinks, query, raw, comp, desc.tokens, desc.first, desc.raw_count, desc.raw_cells,
        desc.raw_start, desc.compressed_count, desc.window, desc.ratio, 64, 512,
        Pointer<const std::int32_t>(desc.positions), Pointer<const std::int32_t>(desc.bank_ids),
        desc.domain == Ds4AttentionDomain::kDecodeHeads ? 0 : desc.compressed_cells,
        Pointer<const ds4_decode_scalars>(desc.decode_scalars),
        Pointer<const ds4_layer_scalars>(desc.layer_scalars), codes, scales, table, diagnostics);
  } else {
    const dim3 grid(desc.tokens, 64, 1);
    attention_decode_mixed_kernel<<<grid, 256, 0, context.stream()>>>(
        output, sinks, query, raw, comp, Pointer<const float>(desc.mask),
        static_cast<std::uint32_t>(!Absent(desc.mask)), desc.tokens, desc.first, desc.raw_count,
        desc.raw_cells, desc.raw_start, desc.compressed_count,
        desc.domain == Ds4AttentionDomain::kDecodeHeads ? 0 : desc.compressed_cells, desc.window,
        desc.ratio, 64, 512, Pointer<const std::int32_t>(desc.positions),
        Pointer<const std::int32_t>(desc.bank_ids),
        Pointer<const ds4_decode_scalars>(desc.decode_scalars),
        Pointer<const ds4_layer_scalars>(desc.layer_scalars), codes, scales,
        Pointer<const std::int32_t>(desc.draft_raw_count), table, diagnostics);
  }
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace

std::expected<void, KernelFailure> PrepareDs4Attention(LaunchContext& launch) {
  if (launch.capturing()) return Rejected("original ds4 attention attributes must precede capture");
  const auto& device = ggml_cuda_info().devices[launch.device()];
  return launch.Run(base::Bytes(0), [&device](auto&) {
    if (device.cc >= 800 && device.smpbo >= kTokentileSmem) {
      CUDA_CHECK(cudaFuncSetAttribute(attention_tokentile_hmma_kernel,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(kTokentileSmem)));
      CUDA_CHECK(cudaFuncSetAttribute(attention_tokentile_union_build_kernel,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize, 65536));
    }
  });
}

std::expected<void, KernelFailure> RunDs4Attention(LaunchContext& launch,
                                                   const Ds4Attention& desc) {
  if (auto checked = CheckDs4Attention(desc); !checked) return checked;
  const auto choice = Choose(launch, desc);
  if (!choice) return std::unexpected(choice.error());
  // Validate the lent handle's original numerical mode before numerical
  // work. Existing native handles use TF32, matching original default ds4;
  // an explicit original quality-mode plan requires DEFAULT_MATH instead.
  if (choice->kind == Ds4AttentionKind::kCublas) {
    cublasMath_t mode = CUBLAS_DEFAULT_MATH;
    const auto status = cublasGetMathMode(launch.cublas()->native(), &mode);
    if (status != CUBLAS_STATUS_SUCCESS) {
      return launch.Run(base::Bytes(0), [status](auto&) { CUBLAS_CHECK(status); });
    }
    const auto expected = desc.quality_mode ? CUBLAS_DEFAULT_MATH : CUBLAS_TF32_TENSOR_OP_MATH;
    if (mode != expected)
      return Rejected("ds4 SGEMM lent handle has a different original math mode");
  }
  return launch.Run(base::Bytes(0), [desc, choice = *choice](auto& context) {
    switch (choice.kind) {
      case Ds4AttentionKind::kTokenTile:
        QueueTokenTile(context, desc, choice.scratch);
        break;
      case Ds4AttentionKind::kHeadGroup:
      case Ds4AttentionKind::kPerHeadSplit:
        QueueSplit(context, desc, choice);
        break;
      case Ds4AttentionKind::kCublas:
        QueueCublas(context, desc, choice.scratch);
        break;
      default:
        if (Indexed(desc))
          QueueIndexed(context, desc, choice);
        else
          QueueOther(context, desc, choice.kind);
        break;
    }
  });
}

std::expected<Ds4AttentionDispatch, KernelFailure> DescribeDs4AttentionDispatch(
    const LaunchContext& launch, const Ds4Attention& desc) {
  if (auto checked = CheckDs4Attention(desc); !checked) return std::unexpected(checked.error());
  const auto choice = Choose(launch, desc);
  if (!choice) return std::unexpected(choice.error());
  return Ds4AttentionDispatch{choice->kind, choice->scratch};
}

std::expected<void, KernelFailure> RunDs4AttentionMask(LaunchContext& launch,
                                                       const Ds4AttentionMask& desc) {
  if (auto checked = CheckDs4AttentionMask(desc); !checked) return checked;
  return launch.Run(base::Bytes(0), [desc](auto& context) {
    const auto elements = std::uint64_t{desc.tokens} * desc.cells;
    topk_mask_kernel<<<static_cast<unsigned>((elements + 255) / 256), 256, 0, context.stream()>>>(
        Pointer<float>(desc.mask), Pointer<const std::uint32_t>(desc.selected), desc.cells,
        desc.tokens, desc.top_k);
    CUDA_CHECK(cudaGetLastError());
  });
}

}  // namespace jitllm::kernels::ggml
