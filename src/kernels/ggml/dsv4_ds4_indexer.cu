// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdint>
#include <cub/block/block_radix_sort.cuh>
#include <expected>
#include <limits>
#include <string>
#include <utility>

#include "base/bytes.h"
#include "common.cuh"
#include "kernels/ggml/dsv4_ds4_indexer.h"
#include "kernels/ggml/ggml_support.h"
#include "kernels/ggml/launch.h"

namespace jitllm::kernels::ggml {
namespace {

// The original kind::mxf4 instructions are device-guarded in this core.
// Native dispatch only selects them on the configured sm_121a device.
#define DS4_CUDA_HAVE_MXF4
#include "kernels/ggml/dsv4_ds4_indexer_core.cuh"
#undef DS4_CUDA_HAVE_MXF4

static_assert(sizeof(ds4_layer_scalars) == sizeof(Ds4IndexerLayerScalars));
using TopkCubSort = cub::BlockRadixSort<std::uint64_t, 512, 16>;

template <typename T>
T* Pointer(const Ds4CacheBuffer& buffer) {
  return reinterpret_cast<T*>(static_cast<std::uintptr_t>(buffer.address));
}

bool Absent(const Ds4CacheBuffer& buffer) { return buffer.address == 0 && buffer.bytes == 0; }

std::unexpected<KernelFailure> Rejected(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}

std::uint64_t Align(std::uint64_t bytes) { return (bytes + 255) & ~std::uint64_t{255}; }

Ds4IndexerScoreKind ClassicKind(const Ds4IndexerScores& desc) {
  if (!Absent(desc.bank_ids) &&
      desc.consecutive_bank == std::numeric_limits<std::uint32_t>::max()) {
    if (!Absent(desc.query_scales)) {
      return desc.tokens >= 2 && desc.tokens <= 8 ? Ds4IndexerScoreKind::kMultisequenceV5e
                                                  : Ds4IndexerScoreKind::kMultisequenceV5d;
    }
    return Ds4IndexerScoreKind::kMultisequenceScalar;
  }
  if (desc.tokens == 1) return Ds4IndexerScoreKind::kDirectOne;
  return desc.quality_mode ? Ds4IndexerScoreKind::kScalar : Ds4IndexerScoreKind::kWmma128;
}

Ds4IndexerSelectKind SelectKind(const Ds4IndexerSelect& desc, bool cub_available) {
  if (desc.kind != Ds4IndexerSelectKind::kOriginalDefault) return desc.kind;
  if (desc.top_k != 512) return Ds4IndexerSelectKind::kInsertion;
  if (desc.score_band <= 1024) return Ds4IndexerSelectKind::kBitonic1024;
  if (desc.score_band <= 2048) return Ds4IndexerSelectKind::kBitonic2048;
  if (desc.score_band <= 4096) {
    return desc.score_band == 4096 && cub_available ? Ds4IndexerSelectKind::kCub8192
                                                    : Ds4IndexerSelectKind::kBitonic4096;
  }
  if (desc.score_band <= 8192)
    return cub_available ? Ds4IndexerSelectKind::kCub8192 : Ds4IndexerSelectKind::kBitonic8192;
  return Ds4IndexerSelectKind::kStream512;
}

void QueueScores(ggml_backend_cuda_context& context, const Ds4IndexerScores& desc,
                 Ds4IndexerScoreKind kind) {
  auto* scores = Pointer<float>(desc.scores);
  const auto* query = Pointer<const float>(desc.query);
  const auto* weights = Pointer<const float>(desc.weights);
  const auto* keys = Pointer<const float>(desc.keys);
  const auto* codes = Pointer<const unsigned char>(desc.key_codes);
  const auto* scales = Pointer<const float>(desc.key_scales);
  const auto* positions = Pointer<const std::int32_t>(desc.positions);
  const auto* banks = Pointer<const std::int32_t>(desc.bank_ids);
  const auto* query_scales = Pointer<const float>(desc.query_scales);
  const auto* live = Pointer<const ds4_layer_scalars>(desc.layer_scalars);
  auto* diagnostics = Pointer<Ds4IndexerDiagnostics>(desc.diagnostics);
  if (desc.consecutive_bank != std::numeric_limits<std::uint32_t>::max()) {
    const auto row0 = static_cast<std::uint64_t>(desc.consecutive_bank) * desc.cells_per_bank;
    keys += row0 * 128;
    if (codes) codes += row0 * 64;
    if (scales) scales += row0 * 4;
    positions = nullptr;
    banks = nullptr;
  }
  switch (kind) {
    case Ds4IndexerScoreKind::kScalar: {
      const dim3 grid(desc.cells, desc.tokens, 1);
      indexer_scores_kernel<<<grid, 256, 0, context.stream()>>>(
          scores, query, weights, keys, desc.cells, desc.tokens, desc.first, 64, 128, desc.ratio,
          desc.scale, static_cast<int>(desc.causal), codes, scales, diagnostics);
      break;
    }
    case Ds4IndexerScoreKind::kDirectOne:
      indexer_score_one_direct_kernel<<<desc.score_band, 128, 0, context.stream()>>>(
          scores, query, weights, keys, desc.cells, desc.first, desc.ratio, desc.scale,
          static_cast<int>(desc.causal), live, codes, scales, diagnostics);
      break;
    case Ds4IndexerScoreKind::kMultisequenceScalar: {
      const dim3 grid(desc.cells, desc.tokens, 1);
      indexer_scores_multiseq_kernel<<<grid, 128, 0, context.stream()>>>(
          scores, query, weights, keys, desc.cells, desc.tokens, desc.first, desc.ratio,
          desc.cells_per_bank, desc.scale, static_cast<int>(desc.causal), positions, banks, codes,
          scales, diagnostics);
      break;
    }
    case Ds4IndexerScoreKind::kMultisequenceV5d: {
      const dim3 grid((desc.cells + 31) / 32, desc.tokens, 1);
      indexer_scores_multiseq_v5d_kernel<<<grid, 128, 0, context.stream()>>>(
          scores, query, weights, keys, desc.cells, desc.tokens, desc.first, desc.ratio,
          desc.cells_per_bank, desc.scale, static_cast<int>(desc.causal), positions, banks, codes,
          scales, query_scales, diagnostics);
      break;
    }
    case Ds4IndexerScoreKind::kMultisequenceV5e: {
      const dim3 grid((desc.cells + 31) / 32, 1, 1);
      indexer_scores_multiseq_v5e_kernel<<<grid, 128, 0, context.stream()>>>(
          scores, query, weights, keys, desc.cells, desc.tokens, desc.first, desc.ratio,
          desc.cells_per_bank, desc.scale, static_cast<int>(desc.causal), positions, banks, codes,
          scales, query_scales, diagnostics);
      break;
    }
    case Ds4IndexerScoreKind::kWmma16: {
      const dim3 grid((desc.cells + 15) / 16, (desc.tokens + 15) / 16, 1);
      indexer_scores_wmma_kernel<<<grid, 32, 0, context.stream()>>>(
          scores, query, weights, keys, desc.cells, desc.tokens, desc.first, 64, 128, desc.ratio,
          desc.scale, static_cast<int>(desc.causal), codes, scales, diagnostics);
      break;
    }
    case Ds4IndexerScoreKind::kWmma32: {
      const dim3 grid((desc.cells + 31) / 32, (desc.tokens + 15) / 16, 1);
      indexer_scores_wmma32_kernel<<<grid, 64, 0, context.stream()>>>(
          scores, query, weights, keys, desc.cells, desc.tokens, desc.first, 64, 128, desc.ratio,
          desc.scale, static_cast<int>(desc.causal), codes, scales, diagnostics);
      break;
    }
    case Ds4IndexerScoreKind::kWmma64: {
      const dim3 grid((desc.cells + 63) / 64, (desc.tokens + 15) / 16, 1);
      indexer_scores_wmma64_kernel<<<grid, 128, 0, context.stream()>>>(
          scores, query, weights, keys, desc.cells, desc.tokens, desc.first, 64, 128, desc.ratio,
          desc.scale, static_cast<int>(desc.causal), codes, scales, diagnostics);
      break;
    }
    case Ds4IndexerScoreKind::kWmma128: {
      const dim3 grid((desc.cells + 127) / 128, (desc.tokens + 31) / 32, 1);
      indexer_scores_wmma128_kernel<<<grid, 256, 0, context.stream()>>>(
          scores, query, weights, keys, desc.cells, desc.tokens, desc.first, 64, 128, desc.ratio,
          desc.scale, static_cast<int>(desc.causal), codes, scales, diagnostics);
      break;
    }
    case Ds4IndexerScoreKind::kMxf4: {
      const auto rows = desc.tokens * 64;
      auto* scratch = Pointer<unsigned char>(desc.scratch);
      const unsigned char* query_codes = nullptr;
      std::uint32_t* query_exponents = nullptr;
      if (!Absent(desc.query_codes)) {
        query_codes = Pointer<const unsigned char>(desc.query_codes);
        query_exponents = reinterpret_cast<std::uint32_t*>(scratch);
        rr_q_scale_pack_kernel<<<(rows + 255) / 256, 256, 0, context.stream()>>>(
            query_scales, query_exponents, rows);
      } else {
        query_codes = scratch;
        query_exponents =
            reinterpret_cast<std::uint32_t*>(scratch + Align(std::uint64_t{rows} * 64));
        rr_q_quant_kernel<<<(rows * 32 + 255) / 256, 256, 0, context.stream()>>>(
            query, scratch, query_exponents, rows);
      }
      CUDA_CHECK(cudaGetLastError());
      const dim3 grid((desc.cells + 255) / 256, (desc.tokens + 15) / 16, 1);
      rr_coarse_mxf4_v3_kernel<<<grid, 512, 0, context.stream()>>>(
          scores, query_codes, query_exponents, weights, desc.cells, desc.tokens, desc.first,
          desc.ratio, desc.scale, codes, scales, desc.cells);
      break;
    }
    default:
      // Host validation/resolution excludes this path.
      break;
  }
  CUDA_CHECK(cudaGetLastError());
}

void QueueSelect(ggml_backend_cuda_context& context, const Ds4IndexerSelect& desc,
                 Ds4IndexerSelectKind kind) {
  auto* selected = Pointer<std::uint32_t>(desc.selected);
  const auto* scores = Pointer<const float>(desc.scores);
  const auto* live = Pointer<const ds4_layer_scalars>(desc.layer_scalars);
  auto* diagnostics = Pointer<Ds4IndexerDiagnostics>(desc.diagnostics);
  switch (kind) {
    case Ds4IndexerSelectKind::kInsertion:
      indexer_topk_kernel<<<desc.tokens, 1, 0, context.stream()>>>(
          selected, scores, desc.score_band, desc.tokens, desc.top_k, live);
      break;
    case Ds4IndexerSelectKind::kBitonic1024:
      indexer_topk_1024_kernel<<<desc.tokens, 1024, 0, context.stream()>>>(
          selected, scores, desc.score_band, desc.tokens, desc.top_k, live);
      break;
    case Ds4IndexerSelectKind::kBitonic2048:
      indexer_topk_pow2_kernel<2048><<<desc.tokens, 1024, 0, context.stream()>>>(
          selected, scores, desc.score_band, desc.tokens, desc.top_k, live);
      break;
    case Ds4IndexerSelectKind::kBitonic4096:
      indexer_topk_pow2_kernel<4096><<<desc.tokens, 1024, 0, context.stream()>>>(
          selected, scores, desc.score_band, desc.tokens, desc.top_k, live);
      break;
    case Ds4IndexerSelectKind::kCub8192:
      indexer_topk_8192_cub_kernel<<<desc.tokens, 512, sizeof(TopkCubSort::TempStorage),
                                     context.stream()>>>(selected, scores, desc.score_band,
                                                         desc.tokens, desc.top_k, live);
      break;
    case Ds4IndexerSelectKind::kBitonic8192:
      indexer_topk_pow2_u16_kernel<8192><<<desc.tokens, 1024, 0, context.stream()>>>(
          selected, scores, desc.score_band, desc.tokens, desc.top_k, live);
      break;
    case Ds4IndexerSelectKind::kStream512:
      indexer_topk_stream512_kernel<<<desc.tokens, 512, 0, context.stream()>>>(
          selected, scores, desc.score_band, desc.tokens, desc.top_k, live, diagnostics);
      break;
    case Ds4IndexerSelectKind::kChunkTree: {
      auto sets = (desc.score_band + 4095) / 4096;
      auto stride = sets * desc.top_k;
      auto* current = Pointer<std::uint32_t>(desc.scratch);
      const dim3 chunks(desc.tokens, sets, 1);
      indexer_topk_chunk_pow2_kernel<4096><<<chunks, 1024, 0, context.stream()>>>(
          current, scores, desc.score_band, desc.tokens, desc.top_k, stride, live);
      CUDA_CHECK(cudaGetLastError());
      while (sets > 8) {
        const auto next_sets = (sets + 7) / 8;
        const auto next_stride = next_sets * desc.top_k;
        auto* next = current + static_cast<std::uint64_t>(desc.tokens) * stride;
        const dim3 merge(desc.tokens, next_sets, 1);
        indexer_topk_tree_merge_pow2_kernel<4096><<<merge, 1024, 0, context.stream()>>>(
            next, current, scores, desc.score_band, desc.tokens, desc.top_k, sets, 8, stride,
            next_stride, live);
        CUDA_CHECK(cudaGetLastError());
        current = next;
        sets = next_sets;
        stride = next_stride;
      }
      indexer_topk_merge_pow2_kernel<4096><<<desc.tokens, 1024, 0, context.stream()>>>(
          selected, current, scores, desc.score_band, desc.tokens, desc.top_k, sets * desc.top_k,
          stride, live);
      break;
    }
    default:
      break;
  }
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace

std::expected<void, KernelFailure> PrepareDs4Indexer(LaunchContext& launch) {
  if (launch.capturing()) return Rejected("ds4 indexer attributes must be prepared before capture");
  const auto& device = ggml_cuda_info().devices[launch.device()];
  return launch.Run(base::Bytes(0), [&device](auto&) {
    CUDA_CHECK(cudaFuncSetAttribute(indexer_scores_multiseq_v5d_kernel,
                                    cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    CUDA_CHECK(cudaFuncSetAttribute(indexer_scores_multiseq_v5e_kernel,
                                    cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    if (device.smpbo >= sizeof(TopkCubSort::TempStorage)) {
      CUDA_CHECK(cudaFuncSetAttribute(indexer_topk_8192_cub_kernel,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(sizeof(TopkCubSort::TempStorage))));
    }
  });
}

std::expected<void, KernelFailure> RunDs4IndexerScores(LaunchContext& launch,
                                                       const Ds4IndexerScores& desc) {
  if (auto checked = CheckDs4IndexerScores(desc); !checked) return checked;
  const auto kind =
      desc.kind == Ds4IndexerScoreKind::kOriginalDefault ? ClassicKind(desc) : desc.kind;
  const auto& device = ggml_cuda_info().devices[launch.device()];
  if (kind == Ds4IndexerScoreKind::kMxf4 && (launch.capturing() || device.cc != 1210))
    return Rejected("original ds4 MXF4 score chain needs eager sm_121a");
  if (device.cc < 800 && kind != Ds4IndexerScoreKind::kScalar &&
      kind != Ds4IndexerScoreKind::kDirectOne && kind != Ds4IndexerScoreKind::kMultisequenceScalar)
    return Rejected("original ds4 indexer WMMA/staging needs sm_80 or later");
  return launch.Run(base::Bytes(0),
                    [desc, kind](auto& context) { QueueScores(context, desc, kind); });
}

std::expected<void, KernelFailure> RunDs4IndexerSelect(LaunchContext& launch,
                                                       const Ds4IndexerSelect& desc) {
  if (auto checked = CheckDs4IndexerSelect(desc); !checked) return checked;
  if (desc.kind == Ds4IndexerSelectKind::kChunkTree && launch.capturing())
    return Rejected("original ds4 tree selector is eager only");
  const auto& device = ggml_cuda_info().devices[launch.device()];
  const bool cub = device.smpbo >= sizeof(TopkCubSort::TempStorage);
  const auto kind = SelectKind(desc, cub);
  if (kind == Ds4IndexerSelectKind::kCub8192 && !cub)
    return Rejected("original ds4 CUB selector exceeds device dynamic shared memory");
  return launch.Run(base::Bytes(0),
                    [desc, kind](auto& context) { QueueSelect(context, desc, kind); });
}

std::expected<Ds4IndexerDispatch, KernelFailure> DescribeDs4IndexerDispatch(
    const LaunchContext& launch, const Ds4IndexerScores& scores, const Ds4IndexerSelect& select) {
  if (auto checked = CheckDs4IndexerScores(scores); !checked)
    return std::unexpected(checked.error());
  if (auto checked = CheckDs4IndexerSelect(select); !checked)
    return std::unexpected(checked.error());
  if (scores.scores.address != select.scores.address || scores.tokens != select.tokens ||
      scores.cells != select.cells || scores.score_band != select.score_band ||
      scores.layer_scalars.address != select.layer_scalars.address ||
      scores.diagnostics.address != select.diagnostics.address)
    return Rejected(
        "ds4 combined scorer/selector must share the original score band and live operands");
  // Selection cannot overwrite immutable scorer inputs, even after their
  // last read on this stream. Views describe the plan's retained resources.
  const Ds4CacheBuffer inputs[]{scores.query,        scores.weights,    scores.keys,
                                scores.key_codes,    scores.key_scales, scores.query_codes,
                                scores.query_scales, scores.positions,  scores.bank_ids,
                                scores.layer_scalars};
  const auto selected_bytes =
      static_cast<std::uint64_t>(select.tokens) * select.top_k * sizeof(std::uint32_t);
  for (const auto& input : inputs) {
    if (!Absent(input) && select.selected.address < input.address + input.bytes &&
        input.address < select.selected.address + selected_bytes)
      return Rejected("ds4 combined selection overlaps a retained score input");
    if (!Absent(input) && !Absent(select.scratch) &&
        select.scratch.address < input.address + input.bytes &&
        input.address < select.scratch.address + select.scratch.bytes)
      return Rejected("ds4 combined selection scratch overlaps a retained score input");
  }
  const auto& device = ggml_cuda_info().devices[launch.device()];
  auto kind =
      scores.kind == Ds4IndexerScoreKind::kOriginalDefault ? ClassicKind(scores) : scores.kind;
  if (scores.kind == Ds4IndexerScoreKind::kOriginalDefault && !launch.capturing() &&
      device.cc == 1210 && select.top_k == 512 && !Absent(scores.scratch)) {
    auto candidate = scores;
    candidate.kind = Ds4IndexerScoreKind::kMxf4;
    if (CheckDs4IndexerScores(candidate)) kind = Ds4IndexerScoreKind::kMxf4;
  }
  if (kind == Ds4IndexerScoreKind::kMxf4 &&
      (launch.capturing() || device.cc != 1210 || select.top_k != 512))
    return Rejected("original ds4 MXF4 score-select needs eager sm_121a and top512");
  if (select.kind == Ds4IndexerSelectKind::kChunkTree && launch.capturing())
    return Rejected("original ds4 tree selector is eager only");
  const bool cub = device.smpbo >= sizeof(TopkCubSort::TempStorage);
  const auto selected_kind = SelectKind(select, cub);
  if (selected_kind == Ds4IndexerSelectKind::kCub8192 && !cub)
    return Rejected("original ds4 CUB selector exceeds device dynamic shared memory");
  if (device.cc < 800 && kind != Ds4IndexerScoreKind::kScalar &&
      kind != Ds4IndexerScoreKind::kDirectOne && kind != Ds4IndexerScoreKind::kMultisequenceScalar)
    return Rejected("original ds4 indexer WMMA/staging needs sm_80 or later");
  return Ds4IndexerDispatch{
      .score_kind = kind,
      .select_kind = selected_kind,
      .cub_temp_storage_bytes = sizeof(TopkCubSort::TempStorage),
      .select_dynamic_shared_bytes =
          selected_kind == Ds4IndexerSelectKind::kCub8192 ? sizeof(TopkCubSort::TempStorage) : 0,
      .device_shared_optin = static_cast<std::uint64_t>(device.smpbo),
      .cub_available = cub};
}

std::expected<void, KernelFailure> RunDs4IndexerScoreSelect(LaunchContext& launch,
                                                            const Ds4IndexerScores& scores,
                                                            const Ds4IndexerSelect& select) {
  const auto dispatch = DescribeDs4IndexerDispatch(launch, scores, select);
  if (!dispatch) return std::unexpected(dispatch.error());
  return launch.Run(base::Bytes(0), [scores, select, dispatch = *dispatch](auto& context) {
    QueueScores(context, scores, dispatch.score_kind);
    if (internal::CudaErrorPending()) return;
    QueueSelect(context, select, dispatch.select_kind);
  });
}

}  // namespace jitllm::kernels::ggml
