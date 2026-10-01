// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <expected>
#include <limits>

#include "base/bytes.h"
#include "common.cuh"
#include "kernels/ggml/dsv4_ds4_comp.h"
#include "kernels/ggml/dsv4_ds4_hc.h"
#include "kernels/ggml/launch.h"

namespace jitllm::kernels::ggml {
namespace {

#include "kernels/ggml/dsv4_ds4_comp_core.cuh"

template <typename T>
T* Pointer(const Ds4CacheBuffer& buffer) {
  return reinterpret_cast<T*>(static_cast<std::uintptr_t>(buffer.address));
}
std::uint32_t Head(const Ds4CompState& state) {
  return state.kind == Ds4CacheKind::kKv512 ? 512U : 128U;
}
std::uint32_t Width(const Ds4CompState& state) {
  return Head(state) * (state.ratio == 4 ? 2U : 1U);
}
std::uint64_t StateElements(const Ds4CompState& state) {
  return static_cast<std::uint64_t>(state.ratio == 4 ? 8U : 128U) * Width(state);
}
std::uint32_t Blocks(std::uint64_t elements) {
  return static_cast<std::uint32_t>((elements + 255) / 256);
}
Ds4CacheBuffer View(const Ds4CacheBuffer& buffer, std::uint64_t offset, std::uint64_t bytes) {
  return {buffer.address + offset, bytes};
}
std::uint32_t ApeType(Ds4CompApe format) { return format == Ds4CompApe::kF16 ? 1U : 0U; }

std::expected<void, KernelFailure> Reset(LaunchContext& launch, const Ds4CompState& state,
                                         bool boot) {
  return launch.Run(base::Bytes(0), [state, boot](auto& context) {
    const auto elements = StateElements(state);
    if (boot) {
      fill_f32_kernel<<<Blocks(elements), 256, 0, context.stream()>>>(Pointer<float>(state.kv),
                                                                      elements, 0);
    } else {
      CUDA_CHECK(cudaMemsetAsync(Pointer<float>(state.kv), 0, elements * 4, context.stream()));
    }
    fill_f32_kernel<<<Blocks(elements), 256, 0, context.stream()>>>(
        Pointer<float>(state.score), elements,
        boot ? kDs4CompBootScore : -std::numeric_limits<float>::infinity());
    CUDA_CHECK(cudaGetLastError());
  });
}

std::expected<void, KernelFailure> SetRows(LaunchContext& launch, const Ds4CompState& state,
                                           const Ds4CacheBuffer& kv, const Ds4CacheBuffer& score,
                                           const Ds4CacheBuffer& ape, Ds4CompApe format,
                                           std::uint32_t first, std::uint32_t source,
                                           std::uint32_t destination, std::uint32_t rows) {
  return launch.Run(base::Bytes(0), [=](auto& context) {
    compressor_set_rows_kernel<<<Blocks(static_cast<std::uint64_t>(rows) * Width(state)), 256, 0,
                                 context.stream()>>>(
        Pointer<float>(state.kv), Pointer<float>(state.score), Pointer<const float>(kv),
        Pointer<const float>(score), Pointer<const void>(ape), 0, ApeType(format), Width(state),
        state.ratio, first, source, destination, rows);
    CUDA_CHECK(cudaGetLastError());
  });
}

std::expected<void, KernelFailure> NormRope(LaunchContext& launch, const Ds4CompChunk& desc,
                                            const Ds4CacheBuffer& values, std::uint32_t rows,
                                            std::uint32_t first) {
  const Ds4Rms norm{.source = values,
                    .weights = desc.norm,
                    .values = values,
                    .width = Head(desc.state),
                    .rows = rows,
                    .epsilon = desc.rms_epsilon};
  if (auto normalized = RunDs4Rms(launch, norm); !normalized) return normalized;
  return launch.Run(base::Bytes(0), [desc, values, rows, first](auto& context) {
    const auto& rope = desc.rope;
    rope_tail_kernel<<<Blocks(static_cast<std::uint64_t>(rows) * 32), 256, 0, context.stream()>>>(
        Pointer<float>(values), rows, 1, Head(desc.state), 64, first, nullptr, desc.state.ratio,
        rope.original_context, 0, rope.frequency_base, rope.frequency_scale, rope.extension,
        rope.attention_factor, rope.beta_fast, rope.beta_slow);
    CUDA_CHECK(cudaGetLastError());
  });
}

std::expected<void, KernelFailure> Qat(LaunchContext& launch, const Ds4CompChunk& desc,
                                       const Ds4CacheBuffer& values, std::uint32_t row,
                                       std::uint32_t rows) {
  const auto code_bytes =
      desc.state.kind == Ds4CacheKind::kKv512 ? kDs4KvPackedRowBytes : kDs4IndexerPackedRowBytes;
  const auto scale_bytes =
      desc.state.kind == Ds4CacheKind::kKv512 ? kDs4KvScaleRowBytes : kDs4IndexerScaleRowBytes;
  const bool packed = desc.codes.address != 0;
  const Ds4CacheQat qat{
      .kind = desc.state.kind,
      .values = values,
      .codes = packed ? View(desc.codes, static_cast<std::uint64_t>(row) * code_bytes,
                             static_cast<std::uint64_t>(rows) * code_bytes)
                      : Ds4CacheBuffer{},
      .scales = packed ? View(desc.scales, static_cast<std::uint64_t>(row) * scale_bytes,
                              static_cast<std::uint64_t>(rows) * scale_bytes)
                       : Ds4CacheBuffer{},
      .rows = rows};
  return RunDs4CacheQat(launch, qat);
}

std::unexpected<KernelFailure> CaptureRefusal() {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected,
                    .detail = "ds4 compressor position/emit program is not graph-replayable"});
}

}  // namespace

std::expected<void, KernelFailure> RunDs4CompInitialize(LaunchContext& launch,
                                                        const Ds4CompState& desc) {
  if (auto checked = CheckDs4CompState(desc); !checked) return checked;
  return Reset(launch, desc, true);
}

std::expected<void, KernelFailure> RunDs4CompPool(LaunchContext& launch, const Ds4CompPool& desc) {
  if (auto checked = CheckDs4CompPool(desc); !checked) return checked;
  return launch.Run(base::Bytes(0), [desc](auto& context) {
    const auto rows = desc.tokens / desc.state.ratio;
    dim3 grid(Blocks(Head(desc.state)), rows, 1);
    compressor_prefill_pool_kernel<<<grid, 256, 0, context.stream()>>>(
        Pointer<float>(desc.values), Pointer<const float>(desc.kv),
        Pointer<const float>(desc.score), Pointer<const float>(desc.state.kv),
        Pointer<const float>(desc.state.score), Pointer<const void>(desc.ape), 0,
        ApeType(desc.ape_format), Head(desc.state), desc.state.ratio, desc.first, rows,
        desc.state.ratio == 4 && desc.first != 0 ? 1U : 0U);
    CUDA_CHECK(cudaGetLastError());
  });
}

std::expected<void, KernelFailure> RunDs4CompRefresh(LaunchContext& launch,
                                                     const Ds4CompRefresh& desc) {
  if (auto checked = CheckDs4CompRefresh(desc); !checked) return checked;
  if (launch.capturing()) return CaptureRefusal();
  if (auto reset = Reset(launch, desc.state, false); !reset) return reset;
  return SetRows(launch, desc.state, desc.kv, desc.score, desc.ape, desc.ape_format, desc.first, 0,
                 0, 4);
}

std::expected<void, KernelFailure> RunDs4CompChunk(LaunchContext& launch,
                                                   const Ds4CompChunk& desc) {
  if (auto checked = CheckDs4CompChunk(desc); !checked) return checked;
  if (launch.capturing()) return CaptureRefusal();
  const auto plan = PlanDs4Comp(desc);
  if (!plan) return std::unexpected(plan.error());
  const auto head = Head(desc.state);
  const auto width = Width(desc.state);
  const auto ratio = desc.state.ratio;
  if (plan->path == Ds4CompPath::kRows) {
    std::uint32_t emitted = 0;
    for (std::uint32_t t = 0; t < desc.tokens; ++t) {
      const auto position = desc.first + t;
      const auto kv = View(desc.kv, static_cast<std::uint64_t>(t) * width * 4,
                           static_cast<std::uint64_t>(width) * 4);
      const auto score = View(desc.score, static_cast<std::uint64_t>(t) * width * 4,
                              static_cast<std::uint64_t>(width) * 4);
      if (auto stored =
              launch.Run(base::Bytes(0),
                         [=](auto& context) {
                           compressor_store_kernel<<<Blocks(width), 256, 0, context.stream()>>>(
                               Pointer<const float>(kv), Pointer<const float>(score),
                               Pointer<float>(desc.state.kv), Pointer<float>(desc.state.score),
                               Pointer<const void>(desc.ape), 0, ApeType(desc.ape_format), head,
                               ratio, position, 1, nullptr, nullptr, nullptr, 0);
                           CUDA_CHECK(cudaGetLastError());
                         });
          !stored)
        return stored;
      if ((position + 1) % ratio != 0) continue;
      const auto values = View(desc.values, static_cast<std::uint64_t>(emitted) * head * 4,
                               static_cast<std::uint64_t>(head) * 4);
      if (auto pooled = launch.Run(
              base::Bytes(0),
              [=](auto& context) {
                compressor_update_pool_kernel<<<Blocks(head), 256, 0, context.stream()>>>(
                    Pointer<float>(values), Pointer<const float>(desc.state.kv),
                    Pointer<const float>(desc.state.score), head, ratio, 0, nullptr, nullptr, 0);
                CUDA_CHECK(cudaGetLastError());
              });
          !pooled)
        return pooled;
      if (auto transformed = NormRope(launch, desc, values, 1, position + 1 - ratio); !transformed)
        return transformed;
      // Original update helper shifts before the caller's external QAT.
      if (ratio == 4) {
        if (auto shifted = launch.Run(
                base::Bytes(0),
                [=](auto& context) {
                  compressor_shift_ratio4_kernel<<<Blocks(static_cast<std::uint64_t>(4) * width),
                                                   256, 0, context.stream()>>>(
                      Pointer<float>(desc.state.kv), Pointer<float>(desc.state.score), width,
                      nullptr, 0);
                  CUDA_CHECK(cudaGetLastError());
                });
            !shifted)
          return shifted;
      }
      if (auto quantized = Qat(launch, desc, values, emitted, 1); !quantized) return quantized;
      ++emitted;
    }
    return {};
  }

  const auto cutoff = desc.tokens / ratio * ratio;
  const auto remaining = desc.tokens - cutoff;
  const bool replay = ratio == 4 && plan->path == Ds4CompPath::kAligned;
  if (!replay) {
    if (auto reset = Reset(launch, desc.state, false); !reset) return reset;
    if (ratio == 4 && cutoff >= ratio) {
      if (auto set = SetRows(launch, desc.state, desc.kv, desc.score, desc.ape, desc.ape_format,
                             desc.first, cutoff - ratio, 0, ratio);
          !set)
        return set;
    }
    if (remaining != 0) {
      if (auto set = SetRows(launch, desc.state, desc.kv, desc.score, desc.ape, desc.ape_format,
                             desc.first, cutoff, ratio == 4 ? ratio : 0U, remaining);
          !set)
        return set;
    }
  }
  if (plan->emitted != 0) {
    const Ds4CompPool pool{.state = desc.state,
                           .kv = desc.kv,
                           .score = desc.score,
                           .ape = desc.ape,
                           .values = desc.values,
                           .ape_format = desc.ape_format,
                           .first = desc.first,
                           .tokens = cutoff};
    if (auto pooled = RunDs4CompPool(launch, pool); !pooled) return pooled;
    const auto values = View(desc.values, 0, static_cast<std::uint64_t>(plan->emitted) * head * 4);
    if (auto transformed = NormRope(launch, desc, values, plan->emitted, desc.first); !transformed)
      return transformed;
    if (desc.state.kind == Ds4CacheKind::kKv512) {
      if (auto quantized = Qat(launch, desc, values, 0, plan->emitted); !quantized)
        return quantized;
    }
  }
  if (replay) {
    if (auto reset = Reset(launch, desc.state, false); !reset) return reset;
    if (auto set = SetRows(launch, desc.state, desc.kv, desc.score, desc.ape, desc.ape_format,
                           desc.first, desc.tokens - ratio, 0, ratio);
        !set)
      return set;
  }
  // Original indexer caller performs its QAT after the CUDA helper's
  // state reset/set, whereas KV FP8 packing is inside that helper.
  if (plan->emitted != 0 && desc.state.kind == Ds4CacheKind::kIndexer128) {
    const auto values = View(desc.values, 0, static_cast<std::uint64_t>(plan->emitted) * head * 4);
    if (auto quantized = Qat(launch, desc, values, 0, plan->emitted); !quantized) return quantized;
  }
  // Caller executes the mandatory small projections + RunRefresh when the
  // scalar Plan marks refresh_required. This preserves original work order.
  return {};
}

}  // namespace jitllm::kernels::ggml
