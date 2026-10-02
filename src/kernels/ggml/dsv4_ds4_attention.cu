// SPDX-FileCopyrightText: 2026 The ds4.c authors
// SPDX-FileCopyrightText: 2026 Entrpi <entrpi@proton.me> (batched-serving fork modifications)
// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// attention_tokentile_hmma_q16_kernel and its F16 Q loader are MIT: a copy
// of Entrpi/ds4 76d51ef82a81b70b78e51a3a6ea11946286de976's token-tile HCA
// core (ds4_cuda.cu, as the locked cuda/attention/ds4_attn_tokentile.cu
// extracts it) that reads F16 Q rows; everything else here is jitLLM's.

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

// clang-format off

// Experimental (docs/experiments/ds4-prefill-stages): the token-tile HCA
// core reading F16 Q rows. tt_load_score_q_frag rounds each F32 Q pair with
// __floats2half2_rn; the Q-head has stored exactly those halves.
__device__ __forceinline__ void tt_load_score_q_frag_h(
        uint32_t (&q_frag)[kTTScoreKStepsPerQuarter][4],
        const half * __restrict__ q,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t tile_base,
        uint32_t head_base) {
    constexpr uint32_t kMtiles = kTTM / 16u;
    constexpr uint32_t kScoreWarps = kMtiles * kTTScoreKQuarters;
    const uint32_t warp = tt_warp_id();
    if (warp >= kScoreWarps) {
        return;
    }
    const uint32_t mtile = warp >> 2u;
    const uint32_t kq = warp & 3u;
    const uint32_t k_base = kq * kTTScoreKSliceDim;
    const uint32_t lane = tt_lane_id();
    const uint32_t a_group = lane >> 2u;
    const uint32_t a_col_pair = (lane & 3u) << 1u;
#pragma unroll
    for (uint32_t kt = 0; kt < kTTScoreKStepsPerQuarter; ++kt) {
        const uint32_t k0 = k_base + kt * 16u;
#pragma unroll
        for (uint32_t r = 0; r < 4u; ++r) {
            const uint32_t m = mtile * 16u + a_group + ((r & 1u) ? 8u : 0u);
            const uint32_t tok = m / kTTG;
            const uint32_t h = m - tok * kTTG;
            const uint32_t gt = tile_base + tok;
            const uint32_t gh = head_base + h;
            const uint32_t d = k0 + a_col_pair + ((r & 2u) ? 8u : 0u);
            half2 packed = __floats2half2_rn(0.0f, 0.0f);
            if (gt < n_tokens && gh < n_head) {
                const half *q_row = q + ((uint64_t)gt * n_head + gh) * kTTHeadDim;
                packed = *reinterpret_cast<const half2 *>(q_row + d);
            }
            q_frag[kt][r] =
                (uint32_t)__half_as_ushort(__low2half(packed)) |
                ((uint32_t)__half_as_ushort(__high2half(packed)) << 16);
        }
    }
}

__global__ static void __launch_bounds__(512, 1) attention_tokentile_hmma_q16_kernel(
        float *heads,
        const float *sinks,
        const half *q,
        const half *raw_kv,
        const half *comp_kv,
        const int2 *union_records,
        const uint32_t *union_counts,
        uint32_t rec_stride,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t raw_row_min) {
    constexpr uint32_t kKvElems = tt_TokentileLayout<kTTStageRows>::ring_plane_elems;
    constexpr uint32_t kProbStride = tt_TokentileLayout<kTTStageRows>::prob_stride;
    const uint32_t tid = threadIdx.x;
    const uint32_t tile_idx = blockIdx.x;
    const uint32_t head_group = blockIdx.y;
    const uint32_t tile_base = tile_idx * kTTTileTokens;
    const uint32_t head_base = head_group * kTTG;
    if (tile_base >= n_tokens) {
        return;
    }
    const uint32_t tile_count =
        n_tokens - tile_base < kTTTileTokens ? n_tokens - tile_base : kTTTileTokens;
    const uint32_t raw_union_count = tile_count + kTTRawWindow - 1u;
    const uint32_t comp_union_count = union_counts[tile_idx];
    const uint32_t n_score = raw_union_count + comp_union_count;
    const uint64_t union_tile_off = (uint64_t)tile_idx * rec_stride;
    const int2 * __restrict__ union_records_tile = union_records + union_tile_off;
    const float score_scale = rsqrtf((float)kTTHeadDim);

    extern __shared__ unsigned char smem[];
    unsigned char *p = tt_align16(smem);
    half *kv_h = reinterpret_cast<half *>(p);
    p = tt_align16(p + 2u * tt_TokentileLayout<kTTStageRows>::ring_plane_bytes);
    half *probs = reinterpret_cast<half *>(p);
    p = tt_align16(p + 2u * kTTM * kProbStride * sizeof(half));
    float4 *score_scratch = reinterpret_cast<float4 *>(p);
    p = tt_align16(p + kTTM * kTTStageRows * sizeof(float4));
    float *max_s = reinterpret_cast<float *>(p);
    p = tt_align16(p + kTTM * sizeof(float));
    float *sum_s = reinterpret_cast<float *>(p);
    p = tt_align16(p + kTTM * sizeof(float));
    float *stage_rescale = reinterpret_cast<float *>(p);
    p = tt_align16(p + kTTM * sizeof(float));
    float *final_scale = reinterpret_cast<float *>(p);
    p = tt_align16(p + kTTM * sizeof(float));
    int2 *rec_ring = reinterpret_cast<int2 *>(p);
    p = tt_align16(p + kTTRecordRingPlanes * kTTStageRows * sizeof(int2));

    for (uint32_t m = tid; m < kTTM; m += blockDim.x) {
        max_s[m] = -INFINITY;
        sum_s[m] = 0.0f;
        stage_rescale[m] = 1.0f;
        final_scale[m] = 0.0f;
    }
    __syncthreads();

    union tt_TokentileRoleRegs {
        uint32_t score_q_frag[kTTScoreKStepsPerQuarter][4];
        float o_acc[2u * kTTTileTokens * kTTG];
    };
    tt_TokentileRoleRegs role_regs;
    tt_load_score_q_frag_h(
        role_regs.score_q_frag, q, n_tokens, n_head, tile_base, head_base);
    if (tt_warp_id() >= 8u) {
#pragma unroll
        for (uint32_t i = 0; i < 2u * kTTM; ++i) {
            role_regs.o_acc[i] = 0.0f;
        }
    }

    uint32_t cur = 0u;
    uint32_t free = 1u;
    if (n_score != 0u) {
        const uint32_t nr0 = n_score < kTTStageRows ? n_score : kTTStageRows;
        tt_issue_kv_stage_cp_async<kTTStageRows, false>(
            kv_h + cur * kKvElems,
            rec_ring,
            0u,
            nr0,
            raw_union_count,
            union_records_tile,
            tile_base,
            raw_kv,
            comp_kv,
            tid);
        tt_issue_record_stage_cp_async<kTTStageRows>(
            rec_ring,
            0u,
            nr0,
            raw_union_count,
            union_records_tile);
        if (kTTStageRows < n_score) {
            const uint32_t nr1 =
                n_score - kTTStageRows < kTTStageRows ? n_score - kTTStageRows : kTTStageRows;
            tt_issue_record_stage_cp_async<kTTStageRows>(
                rec_ring + kTTStageRows,
                kTTStageRows,
                nr1,
                raw_union_count,
                union_records_tile);
        }
        tt_cp_async_commit();
        tt_cp_async_wait_group<0>();
    }
    __syncthreads();

    uint32_t prob_cur = 0u;
    for (uint32_t row0 = 0u; row0 < n_score; row0 += kTTStageRows) {
        const uint32_t nr = n_score - row0 < kTTStageRows ? n_score - row0 : kTTStageRows;
        half *kv_cur = kv_h + cur * kKvElems;
        half *kv_free = kv_h + free * kKvElems;

        tt_hmma_score_stage<kTTStageRows>(
            score_scratch, role_regs.score_q_frag, kv_cur, nr, score_scale);
        if (row0 != 0u) {
            const half *kv_prev = kv_h + (cur ^ 1u) * kKvElems;
            const half *probs_prev = probs + (prob_cur ^ 1u) * kTTM * kProbStride;
            tt_pv_mma_stage<kTTStageRows>(
                role_regs.o_acc, probs_prev, stage_rescale, kv_prev);
        }
        __syncthreads();

        const uint32_t next_row0 = row0 + kTTStageRows;
        const bool has_next = next_row0 < n_score;
        if (has_next) {
            const uint32_t next_nr =
                n_score - next_row0 < kTTStageRows ? n_score - next_row0 : kTTStageRows;
            tt_issue_kv_stage_cp_async<kTTStageRows, true>(
                kv_free,
                rec_ring + (((row0 / kTTStageRows) + 1u) & 3u) * kTTStageRows,
                next_row0,
                next_nr,
                raw_union_count,
                union_records_tile,
                tile_base,
                raw_kv,
                comp_kv,
                tid);
            const uint32_t prefetch_row0 = next_row0 + kTTStageRows;
            if (prefetch_row0 < n_score) {
                const uint32_t prefetch_nr =
                    n_score - prefetch_row0 < kTTStageRows
                        ? n_score - prefetch_row0
                        : kTTStageRows;
                tt_issue_record_stage_cp_async<kTTStageRows>(
                    rec_ring + (((row0 / kTTStageRows) + 2u) & 3u) * kTTStageRows,
                    prefetch_row0,
                    prefetch_nr,
                    raw_union_count,
                    union_records_tile);
            }
            tt_cp_async_commit();
        }

        half *probs_cur = probs + prob_cur * kTTM * kProbStride;
        tt_softmax_stage<kTTStageRows>(
            probs_cur,
            stage_rescale,
            max_s,
            sum_s,
            score_scratch,
            rec_ring + ((row0 / kTTStageRows) & 3u) * kTTStageRows,
            row0,
            nr,
            raw_union_count,
            tile_count,
            tile_base,
            raw_row_min);
        tt_cp_async_wait_group<0>();
        __syncthreads();
        cur ^= 1u;
        free ^= 1u;
        prob_cur ^= 1u;
    }

    if (n_score != 0u) {
        const half *kv_prev = kv_h + (cur ^ 1u) * kKvElems;
        const half *probs_prev = probs + (prob_cur ^ 1u) * kTTM * kProbStride;
        tt_pv_mma_stage<kTTStageRows>(
            role_regs.o_acc, probs_prev, stage_rescale, kv_prev);
    }
    __syncthreads();

    for (uint32_t m = tid; m < kTTM; m += blockDim.x) {
        const uint32_t tok = m / kTTG;
        const uint32_t h = m - tok * kTTG;
        const uint32_t gt = tile_base + tok;
        const uint32_t gh = head_base + h;
        if (gt < n_tokens && gh < n_head) {
            const float sink = sinks[gh];
            const float old_m = max_s[m];
            const float new_m = fmaxf(old_m, sink);
            const float old_scale = old_m == -INFINITY ? 0.0f : expf(old_m - new_m);
            const float sink_scale = expf(sink - new_m);
            const float den = sum_s[m] * old_scale + sink_scale;
            final_scale[m] = den == 0.0f ? 0.0f : old_scale / den;
        } else {
            final_scale[m] = 0.0f;
        }
    }
    __syncthreads();

    tt_pv_mma_epilogue(
        role_regs.o_acc, final_scale, heads, n_tokens, n_head, tile_base, head_base);
}
// clang-format on

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

int Ds4HcaCoreQ16Prepare() {
  return static_cast<int>(cudaFuncSetAttribute(attention_tokentile_hmma_q16_kernel,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                                               static_cast<int>(kTokentileSmem)));
}

int Ds4HcaCoreQ16Launch(float* out, const float* sinks, const void* q, const void* raw,
                        const void* compressed, const void* records, const void* counts,
                        std::uint32_t stride, std::uint32_t tokens, std::uint32_t heads,
                        std::uint32_t raw_row_min, void* stream) {
  if (out == nullptr || sinks == nullptr || q == nullptr || raw == nullptr ||
      compressed == nullptr || records == nullptr || counts == nullptr || tokens == 0 ||
      tokens > 4096 || heads != 64 || stride == 0 || stride > 32768 || raw_row_min > 127 ||
      reinterpret_cast<std::uintptr_t>(out) % 16 != 0 ||
      reinterpret_cast<std::uintptr_t>(q) % 16 != 0 ||
      reinterpret_cast<std::uintptr_t>(raw) % 16 != 0 ||
      reinterpret_cast<std::uintptr_t>(compressed) % 16 != 0 ||
      reinterpret_cast<std::uintptr_t>(records) % 16 != 0 ||
      reinterpret_cast<std::uintptr_t>(counts) % 4 != 0 ||
      reinterpret_cast<std::uintptr_t>(sinks) % 4 != 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const dim3 grid((tokens + 3U) / 4U, heads / kTTG, 1);
  attention_tokentile_hmma_q16_kernel<<<grid, kTTThreads, kTokentileSmem,
                                        static_cast<cudaStream_t>(stream)>>>(
      out, sinks, static_cast<const half*>(q), static_cast<const half*>(raw),
      static_cast<const half*>(compressed), static_cast<const int2*>(records),
      static_cast<const std::uint32_t*>(counts), stride, tokens, heads, raw_row_min);
  return static_cast<int>(cudaGetLastError());
}

}  // namespace jitllm::kernels::ggml
