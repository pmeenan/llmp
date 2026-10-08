// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

#include <algorithm>
#include <climits>
#include <cstdint>

#include "fattn-mma-f16.cuh"
#include "kernels/ggml/fattn_mma.h"
#include "kernels/ggml/fattn_owner.h"
#include "kernels/ggml/fattn_owner_kernel.cuh"
#include "kernels/ggml/fattn_owner_partial_kernel.cuh"
#include "kernels/ggml/validate_util.h"

namespace llmp::kernels::ggml {
namespace {
using detail::Rejected;
using llmp_fattn_owner::flash_attn_owner_f16;
using llmp_fattn_owner::flash_attn_owner_fixup_general;
using llmp_fattn_owner::flash_attn_owner_fixup_uniform;
using llmp_fattn_owner::flash_attn_owner_partial_f16;
using llmp_fattn_owner::OwnerBases;

template <int D, int Columns, int Group, bool Softcap = false, bool Bounded = false>
std::expected<void, KernelFailure> Resources(int device, FlashAttnOwnersPlan& plan) {
  constexpr int cols = Columns * Group;
  const int cc = ggml_cuda_info().devices[device].cc;
  const int threads = ggml_cuda_fattn_mma_get_nthreads(D, D, cols, cc);
  const int batch = ggml_cuda_fattn_mma_get_nbatch_fa(D, D, cols, cc);
  const int k2 = ggml_cuda_fattn_mma_get_nbatch_K2(D, D, cols, cc);
  const int v2 = ggml_cuda_fattn_mma_get_nbatch_V2(D, D, cols, cc);
  const int combine = ggml_cuda_fattn_mma_get_nbatch_combine(D, D, cols, cc);
  const bool q_in_reg = ggml_cuda_fattn_mma_get_Q_in_reg(D, D, cols, cc);
  const int stages = ggml_cuda_fattn_mma_get_nstages(D, D, Columns, Group, cc);
  const int warp = ggml_cuda_info().devices[device].warp_size;
  if (warp != 32 || threads % warp != 0) return Rejected("owner MMA requires the GB10 warp");
  const int warps = threads / warp;
  const bool swizzled = ggml_cuda_fattn_mma_get_swizzled(D, D, cols, cc);
  const int tile_k = swizzled ? k2 : k2 + 4;
  const int tile_v = swizzled ? v2 : v2 + 4;
  const auto size = [](int value) { return static_cast<std::size_t>(value); };
  const std::size_t kv1 = size(batch) * size(std::max(tile_k, tile_v)) * sizeof(half2);
  const std::size_t kv2 = size(batch) * size(tile_k + tile_v) * sizeof(half2);
  const std::size_t q = std::size_t(cols) * (D / 2 + 4) * sizeof(half2);
  const std::size_t mask = size(Columns) * size(batch / 2 + 4) * sizeof(half2);
  const std::size_t join =
      size(warps) * size(std::min(cols, get_cols_per_warp(cc))) * size(combine + 4) * sizeof(half2);
  const auto kv = stages <= 1 ? kv1 : kv2;
  plan.shared_bytes = std::max(join, q_in_reg ? std::max(q, kv + mask) : q + kv + mask);
  plan.threads = threads;
  const auto kernel = flash_attn_owner_f16<D, D, Columns, Group, Softcap, false, false, Bounded>;
  auto status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     static_cast<int>(plan.shared_bytes));
  if (status == cudaSuccess)
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&plan.owner_blocks_per_sm, kernel,
                                                           threads, plan.shared_bytes);
  if (status != cudaSuccess || plan.owner_blocks_per_sm <= 0)
    return std::unexpected(
        KernelFailure{.error = KernelError::kUnknown, .detail = "owner MMA resource query failed"});
  // Deliberately do not use owner occupancy to recompute plan.original.blocks.
  return {};
}

template <int D, int Columns, int Group>
std::expected<void, KernelFailure> ResourcesPartial(int device, FlashAttnOwnersPlan& plan) {
  constexpr int cols = Columns * Group;
  const int cc = ggml_cuda_info().devices[device].cc;
  const int threads = ggml_cuda_fattn_mma_get_nthreads(D, D, cols, cc);
  const int batch = ggml_cuda_fattn_mma_get_nbatch_fa(D, D, cols, cc);
  const int k2 = ggml_cuda_fattn_mma_get_nbatch_K2(D, D, cols, cc);
  const int v2 = ggml_cuda_fattn_mma_get_nbatch_V2(D, D, cols, cc);
  const int combine = ggml_cuda_fattn_mma_get_nbatch_combine(D, D, cols, cc);
  const bool q_in_reg = ggml_cuda_fattn_mma_get_Q_in_reg(D, D, cols, cc);
  const int stages = ggml_cuda_fattn_mma_get_nstages(D, D, Columns, Group, cc);
  const int warp = ggml_cuda_info().devices[device].warp_size;
  if (warp != 32 || threads % warp != 0) return Rejected("owner MMA requires the GB10 warp");
  const int warps = threads / warp;
  const bool swizzled = ggml_cuda_fattn_mma_get_swizzled(D, D, cols, cc);
  const int tile_k = swizzled ? k2 : k2 + 4;
  const int tile_v = swizzled ? v2 : v2 + 4;
  const auto size = [](int value) { return static_cast<std::size_t>(value); };
  const std::size_t kv1 = size(batch) * size(std::max(tile_k, tile_v)) * sizeof(half2);
  const std::size_t kv2 = size(batch) * size(tile_k + tile_v) * sizeof(half2);
  const std::size_t q = std::size_t(cols) * (D / 2 + 4) * sizeof(half2);
  const std::size_t mask = size(Columns) * size(batch / 2 + 4) * sizeof(half2);
  const std::size_t join =
      size(warps) * size(std::min(cols, get_cols_per_warp(cc))) * size(combine + 4) * sizeof(half2);
  const auto kv = stages <= 1 ? kv1 : kv2;
  plan.shared_bytes = std::max(join, q_in_reg ? std::max(q, kv + mask) : q + kv + mask);
  plan.threads = threads;
  const auto kernel = flash_attn_owner_partial_f16<D, D, Columns, Group, false, false, false>;
  auto status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     static_cast<int>(plan.shared_bytes));
  if (status == cudaSuccess)
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&plan.owner_blocks_per_sm, kernel,
                                                           threads, plan.shared_bytes);
  if (status != cudaSuccess || plan.owner_blocks_per_sm <= 0)
    return std::unexpected(
        KernelFailure{.error = KernelError::kUnknown, .detail = "owner MMA resource query failed"});
  // Deliberately do not use owner occupancy to recompute plan.original.blocks.
  return {};
}

template <int D, int Columns, int Group, bool Softcap = false, bool Bounded = false>
void Queue(ggml_backend_cuda_context& ctx, const FlashAttnOwners& in,
           const FlashAttnOwnersPlan& plan) {
  auto stream = ctx.stream();
  ggml_cuda_pool_alloc<int> maximum(ctx.pool());
  ggml_cuda_pool_alloc<float2> metadata(ctx.pool());
  const int heads = static_cast<int>(in.q->ne[2]);
  const int kvheads = heads / Group;
  const int owners = static_cast<int>(in.owner_count);
  const int query_rows = static_cast<int>(in.q->ne[1]);
  const int query_tiles = (query_rows + Columns - 1) / Columns;
  const int cells = static_cast<int>(in.mask->ne[0]);
  const int tiles = query_tiles * kvheads * owners;
  const int batch = ggml_cuda_fattn_mma_get_nbatch_fa(D, D, Columns * Group,
                                                      ggml_cuda_info().devices[ctx.device].cc);
  const int kvtiles = (cells + batch - 1) / batch;
  const auto blocks = static_cast<unsigned>(plan.original.blocks);
  maximum.alloc(static_cast<std::size_t>(query_tiles * owners));
  const ggml_cuda_kernel_launch_params mask_launch(
      dim3(static_cast<unsigned>(query_tiles), static_cast<unsigned>(owners), 1),
      dim3(FATTN_KQ_STRIDE / 2, 1, 1), 0, stream);
  ggml_cuda_kernel_launch(flash_attn_mask_to_KV_max<Columns>, mask_launch,
                          static_cast<const half2*>(in.mask->data), maximum.ptr,
                          cells / FATTN_KQ_STRIDE, in.mask->nb[1] / sizeof(half2),
                          in.mask->nb[3] / sizeof(half2));
  CUDA_CHECK(cudaGetLastError());
  if (tiles % plan.original.blocks != 0)
    metadata.alloc(std::size_t(blocks) * Columns * Group * (2 + D / 2));
  llmp_fattn_owner::OwnerRootBases<Bounded> bases{};
  for (std::size_t owner = 0; owner < in.owner_count; ++owner) {
    bases.k[owner] = static_cast<const char*>(in.k[owner]->data);
    bases.v[owner] = static_cast<const char*>(in.v[owner]->data);
    if constexpr (Bounded) bases.cells[owner] = static_cast<int>(in.k[owner]->ne[1]);
  }
  const auto* k = in.k[0];
  const auto* v = in.v[0];
  const ggml_cuda_kernel_launch_params main_launch(
      dim3(blocks, 1, 1), dim3(32, static_cast<unsigned>(plan.threads / 32), 1), plan.shared_bytes,
      stream);
  // Match the original case's constants and dimensions exactly. ne13 is a
  // logical stream count only; no contiguous K/V sequence span is accessed.
  ggml_cuda_kernel_launch(
      flash_attn_owner_f16<D, D, Columns, Group, Softcap, false, false, Bounded>, main_launch,
      static_cast<const char*>(in.q->data), bases, static_cast<const char*>(in.mask->data),
      static_cast<const char*>(nullptr), maximum.ptr, static_cast<float*>(in.output->data),
      metadata.ptr, Softcap ? 1.0f / static_cast<float>(in.logit_softcap) : 1.0f, 0.0f, 1.0f, 1.0f,
      static_cast<std::uint32_t>(heads), static_cast<float>(in.logit_softcap), D,
      init_fastdiv_values(static_cast<std::uint64_t>(query_rows)), heads, owners,
      static_cast<std::int32_t>(in.q->nb[1]), static_cast<std::int32_t>(in.q->nb[2]),
      static_cast<std::int32_t>(in.q->nb[3]), D, cells, kvheads, owners,
      static_cast<std::int32_t>(k->nb[1]), static_cast<std::int32_t>(k->nb[2]),
      static_cast<std::int64_t>(k->nb[3]), static_cast<std::int32_t>(v->nb[1]),
      static_cast<std::int32_t>(v->nb[2]), static_cast<std::int64_t>(v->nb[3]),
      static_cast<std::int32_t>(in.mask->ne[1]), 1, owners,
      static_cast<std::int32_t>(in.mask->nb[1]), static_cast<std::int32_t>(in.mask->nb[2]),
      static_cast<std::int64_t>(in.mask->nb[3]));
  CUDA_CHECK(cudaGetLastError());
  if (plan.original.blocks % tiles == 0 && plan.original.blocks > tiles) {
    const ggml_cuda_kernel_launch_params fixup(dim3(static_cast<unsigned>(tiles), Columns, Group),
                                               dim3(D, 1, 1), 0, stream);
    ggml_cuda_kernel_launch(flash_attn_stream_k_fixup_uniform<D, Columns, Group>, fixup,
                            static_cast<float*>(in.output->data), metadata.ptr, query_rows, heads,
                            kvheads, plan.original.blocks, Group, plan.original.blocks / tiles,
                            init_fastdiv_values(static_cast<std::uint64_t>(query_tiles * kvheads)),
                            init_fastdiv_values(static_cast<std::uint64_t>(query_tiles)),
                            init_fastdiv_values(static_cast<std::uint64_t>(query_tiles)));
  } else if (tiles % plan.original.blocks != 0) {
    const ggml_cuda_kernel_launch_params fixup(dim3(blocks, Columns, Group), dim3(D, 1, 1), 0,
                                               stream);
    ggml_cuda_kernel_launch(
        flash_attn_stream_k_fixup_general<D, Columns, Group>, fixup,
        static_cast<float*>(in.output->data), metadata.ptr, query_rows, heads, Group,
        kvtiles * tiles,
        init_fastdiv_values(static_cast<std::uint64_t>(kvtiles * query_tiles * kvheads)),
        init_fastdiv_values(static_cast<std::uint64_t>(kvtiles * query_tiles)),
        init_fastdiv_values(static_cast<std::uint64_t>(kvtiles * query_tiles)),
        init_fastdiv_values(static_cast<std::uint64_t>(kvtiles)));
  }
  CUDA_CHECK(cudaGetLastError());
}
template <int D, int Columns, int Group>
void QueuePartial(ggml_backend_cuda_context& ctx, const FlashAttnOwners& in,
                  const FlashAttnOwnersPlan& plan) {
  auto stream = ctx.stream();
  ggml_cuda_pool_alloc<int> maximum(ctx.pool());
  ggml_cuda_pool_alloc<float2> metadata(ctx.pool());
  const int heads = static_cast<int>(in.q->ne[2]);
  const int kvheads = heads / Group;
  const int owners = static_cast<int>(in.owner_count);
  constexpr int query_rows = 1;
  const int cells = static_cast<int>(in.mask->ne[0]);
  const int cohort = static_cast<int>(in.logical_cohort);
  const int tiles = kvheads * cohort;
  const int batch = ggml_cuda_fattn_mma_get_nbatch_fa(D, D, Columns * Group,
                                                      ggml_cuda_info().devices[ctx.device].cc);
  const int kvtiles = (cells + batch - 1) / batch;
  const auto blocks = static_cast<unsigned>(plan.original.blocks);
  maximum.alloc(static_cast<std::size_t>(owners));
  const ggml_cuda_kernel_launch_params mask_launch(dim3(1, static_cast<unsigned>(owners), 1),
                                                   dim3(FATTN_KQ_STRIDE / 2, 1, 1), 0, stream);
  ggml_cuda_kernel_launch(flash_attn_mask_to_KV_max<Columns>, mask_launch,
                          static_cast<const half2*>(in.mask->data), maximum.ptr,
                          cells / FATTN_KQ_STRIDE, in.mask->nb[1] / sizeof(half2),
                          in.mask->nb[3] / sizeof(half2));
  CUDA_CHECK(cudaGetLastError());
  if (tiles % plan.original.blocks != 0)
    metadata.alloc(std::size_t(blocks) * Columns * Group * (2 + D / 2));
  OwnerBases bases{};
  for (std::size_t owner = 0; owner < in.owner_count; ++owner) {
    bases.k[owner] = static_cast<const char*>(in.k[owner]->data);
    bases.v[owner] = static_cast<const char*>(in.v[owner]->data);
  }
  const auto* k = in.k[0];
  const auto* v = in.v[0];
  const ggml_cuda_kernel_launch_params main_launch(
      dim3(blocks, 1, 1), dim3(32, static_cast<unsigned>(plan.threads / 32), 1), plan.shared_bytes,
      stream);
  // Match the original case's constants and dimensions exactly. ne13 is a
  // logical stream count only; no contiguous K/V sequence span is accessed.
  ggml_cuda_kernel_launch(
      flash_attn_owner_partial_f16<D, D, Columns, Group, false, false, false>, main_launch,
      static_cast<const char*>(in.q->data), bases, static_cast<const char*>(in.mask->data),
      static_cast<const char*>(nullptr), maximum.ptr, static_cast<float*>(in.output->data),
      metadata.ptr, 1.0f, 0.0f, 1.0f, 1.0f, static_cast<std::uint32_t>(heads), 0.0f, D,
      init_fastdiv_values(query_rows), heads, cohort, static_cast<std::int32_t>(in.q->nb[1]),
      static_cast<std::int32_t>(in.q->nb[2]), static_cast<std::int32_t>(in.q->nb[3]), D, cells,
      kvheads, cohort, static_cast<std::int32_t>(k->nb[1]), static_cast<std::int32_t>(k->nb[2]),
      static_cast<std::int64_t>(k->nb[3]), static_cast<std::int32_t>(v->nb[1]),
      static_cast<std::int32_t>(v->nb[2]), static_cast<std::int64_t>(v->nb[3]), 32, 1, owners,
      static_cast<std::int32_t>(in.mask->nb[1]), static_cast<std::int32_t>(in.mask->nb[2]),
      static_cast<std::int64_t>(in.mask->nb[3]), static_cast<int>(in.owner_offset), owners);
  CUDA_CHECK(cudaGetLastError());
  if (plan.original.blocks % tiles == 0 && plan.original.blocks > tiles) {
    const ggml_cuda_kernel_launch_params fixup(dim3(static_cast<unsigned>(tiles), Columns, Group),
                                               dim3(D, 1, 1), 0, stream);
    ggml_cuda_kernel_launch(flash_attn_owner_fixup_uniform<D, Columns, Group>, fixup,
                            static_cast<float*>(in.output->data), metadata.ptr, query_rows, heads,
                            kvheads, plan.original.blocks, Group, plan.original.blocks / tiles,
                            init_fastdiv_values(static_cast<std::uint64_t>(kvheads)),
                            init_fastdiv_values(1), init_fastdiv_values(1),
                            static_cast<int>(in.owner_offset), owners);
  } else if (tiles % plan.original.blocks != 0) {
    const ggml_cuda_kernel_launch_params fixup(dim3(blocks, Columns, Group), dim3(D, 1, 1), 0,
                                               stream);
    ggml_cuda_kernel_launch(flash_attn_owner_fixup_general<D, Columns, Group>, fixup,
                            static_cast<float*>(in.output->data), metadata.ptr, query_rows, heads,
                            Group, kvtiles * tiles,
                            init_fastdiv_values(static_cast<std::uint64_t>(kvtiles * kvheads)),
                            init_fastdiv_values(static_cast<std::uint64_t>(kvtiles)),
                            init_fastdiv_values(static_cast<std::uint64_t>(kvtiles)),
                            init_fastdiv_values(static_cast<std::uint64_t>(kvtiles)),
                            static_cast<int>(in.owner_offset), owners);
  }
  CUDA_CHECK(cudaGetLastError());
}
}  // namespace

std::expected<FlashAttnOwnersPlan, KernelFailure> PlanFlashAttnOwners(const LaunchContext& launch,
                                                                      const FlashAttnOwners& in) {
  if (auto checked = CheckFlashAttnOwners(in); !checked) return std::unexpected(checked.error());
  const auto& device = ggml_cuda_info().devices[launch.device()];
  if (device.cc != 1210 || device.nsm <= 0) return Rejected("owner MMA diagnostic is GB10 only");
  const int d = static_cast<int>(in.q->ne[0]);
  const bool prefill = in.q->ne[1] > 1;
  const int columns = d == 512            ? 1
                      : in.q->ne[1] <= 4  ? 4
                      : in.q->ne[1] <= 8  ? 8
                      : in.q->ne[1] <= 16 ? 16
                                          : 32;
  const int query_tiles = (static_cast<int>(in.q->ne[1]) + columns - 1) / columns;
  auto original =
      d == 256 ? detail::FlashAttnMmaShapeGqa2(columns, launch.device(), in.logit_softcap != 0)
               : detail::FlashAttnMmaShape512(1, false, launch.device());
  if (!original)
    return std::unexpected(
        KernelFailure{.error = KernelError::kUnknown, .detail = original.error()});
  FlashAttnOwnersPlan plan;
  plan.original_blocks_per_sm = original->blocks_per_sm;
  auto& geometry = plan.original;
  geometry.head = d;
  geometry.columns = columns;
  geometry.group = d == 256 ? 2 : 8;
  geometry.mask_prepass = true;
  const bool partial = detail::PartialOwnerCohort(in.logical_cohort);
  const int tiles = query_tiles * static_cast<int>(in.k[0]->ne[2]) *
                    static_cast<int>(partial ? in.logical_cohort : in.owner_count);
  const int cells = static_cast<int>(in.mask->ne[0]);
  const int kvtiles = (cells + original->kv_batch - 1) / original->kv_batch;
  if (device.nsm > INT_MAX / original->blocks_per_sm)
    return Rejected("owner MMA occupancy grid exceeds the launcher bounds");
  auto partition = detail::PlanOwnerPartition(
      original->blocks_per_sm * device.nsm, kvtiles, query_tiles * static_cast<int>(in.k[0]->ne[2]),
      in.logical_cohort, original->async_kv_preload, prefill && in.logit_softcap == 0,
      prefill ? query_tiles : 0);
  if (!partition) return std::unexpected(partition.error());
  plan.cohort_blocks = partition->cohort_blocks;
  plan.effective_cohort = partition->effective_cohort;
  geometry.blocks = partition->quad_blocks;
  geometry.scratch = static_cast<std::uint64_t>(query_tiles) * in.owner_count * sizeof(int);
  if (tiles % geometry.blocks != 0)
    geometry.scratch = 256 + std::uint64_t(geometry.blocks) * std::uint64_t(geometry.columns) *
                                 std::uint64_t(geometry.group) * std::uint64_t(2 + d / 2) *
                                 sizeof(float2);
  const auto prefill_resources = [&]<int Columns>() {
    return in.logit_softcap != 0 ? Resources<256, Columns, 2, true>(launch.device(), plan)
                                 : Resources<256, Columns, 2, false>(launch.device(), plan);
  };
  const auto checked =
      prefill ? (columns == 4    ? prefill_resources.template operator()<4>()
                 : columns == 8  ? prefill_resources.template operator()<8>()
                 : columns == 16 ? prefill_resources.template operator()<16>()
                                 : prefill_resources.template operator()<32>())
      : in.bounded_roots
          ? (in.logit_softcap != 0
                 ? Resources<256, 4, 2, true, true>(launch.device(), plan)
                 : (d == 256 ? Resources<256, 4, 2, false, true>(launch.device(), plan)
                             : Resources<512, 1, 8, false, true>(launch.device(), plan)))
      : in.logit_softcap != 0 ? Resources<256, 4, 2, true>(launch.device(), plan)
      : partial               ? (d == 256 ? ResourcesPartial<256, 4, 2>(launch.device(), plan)
                                          : ResourcesPartial<512, 1, 8>(launch.device(), plan))
                              : (d == 256 ? Resources<256, 4, 2>(launch.device(), plan)
                                          : Resources<512, 1, 8>(launch.device(), plan));
  if (!checked) return std::unexpected(checked.error());
  return plan;
}

std::expected<void, KernelFailure> FlashAttnOwnerRoots(LaunchContext& launch,
                                                       const FlashAttnOwners& in) {
  auto plan = PlanFlashAttnOwners(launch, in);
  if (!plan) return std::unexpected(plan.error());
  return launch.Run(base::Bytes(plan->original.scratch), [&](ggml_backend_cuda_context& context) {
    if (in.q->ne[1] > 1) {
      const auto queue = [&]<int Columns>() {
        if (in.logit_softcap != 0)
          Queue<256, Columns, 2, true>(context, in, *plan);
        else
          Queue<256, Columns, 2, false>(context, in, *plan);
      };
      if (plan->original.columns == 4)
        queue.template operator()<4>();
      else if (plan->original.columns == 8)
        queue.template operator()<8>();
      else if (plan->original.columns == 16)
        queue.template operator()<16>();
      else
        queue.template operator()<32>();
    } else if (detail::PartialOwnerCohort(in.logical_cohort)) {
      if (plan->original.head == 256)
        QueuePartial<256, 4, 2>(context, in, *plan);
      else
        QueuePartial<512, 1, 8>(context, in, *plan);
    } else if (in.bounded_roots) {
      if (in.logit_softcap != 0)
        Queue<256, 4, 2, true, true>(context, in, *plan);
      else if (plan->original.head == 256)
        Queue<256, 4, 2, false, true>(context, in, *plan);
      else
        Queue<512, 1, 8, false, true>(context, in, *plan);
    } else if (in.logit_softcap != 0)
      Queue<256, 4, 2, true>(context, in, *plan);
    else if (plan->original.head == 256)
      Queue<256, 4, 2>(context, in, *plan);
    else
      Queue<512, 1, 8>(context, in, *plan);
  });
}
}  // namespace llmp::kernels::ggml
