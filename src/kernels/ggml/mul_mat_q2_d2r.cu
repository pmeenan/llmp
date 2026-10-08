// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
// The native preparation/scatter layout used by mul_mat_q.cu feeds one
// existing ds4 D2R product. Cache, weights and route weighting are unchanged.
#include <cstdint>
#include <expected>
#include <string>
#include <utility>

#include "base/bytes.h"
#include "common.cuh"
#include "cuda/mmq/ds4_mmq_d2r.cuh"
#include "kernels/ggml/ggml_support.h"
#include "kernels/ggml/ops_ext.h"
#include "kernels/ggml/validate_ext.h"
#include "mmid.cuh"
#include "mmq.cuh"
#include "quantize.cuh"

namespace llmp::kernels::ggml {
namespace {
auto Rejected(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}
struct Draws {
  std::uint64_t bytes = 0;
  void Add(std::uint64_t n) { bytes = ((bytes + 255) / 256) * 256 + n; }
};
}  // namespace

bool MulMatIdQ2D2rFits(const LaunchContext& launch, const ggml_tensor* node) {
  if (!CheckMulMatIdQ2D2r(node)) return false;
  const auto* w = node->src[0];
  const auto* x = node->src[1];
  return ggml_cuda_info().devices[launch.device()].cc == 1210 && w->ne[0] == 2048 &&
         w->ne[1] == 4096 && w->ne[2] == 256 && x->ne[2] >= kDsv4StageMinRows &&
         x->ne[2] <= kDsv4StageMaxRows && node->src[2]->ne[0] == 6;
}

std::expected<std::uint64_t, KernelFailure> PlanMulMatIdQ2D2r(const LaunchContext& launch,
                                                              const ggml_tensor* node) {
  if (auto check = CheckMulMatIdQ2D2r(node); !check) return std::unexpected(check.error());
  const auto* w = node->src[0];
  const auto* x = node->src[1];
  const auto* ids = node->src[2];
  const auto rows = x->ne[2] * ids->ne[0];
  const int cc = ggml_cuda_info().devices[launch.device()].cc;
  if (!ds4_mmq_q2_K_moe_d2r_available(cc) ||
      static_cast<std::uint64_t>(x->ne[2]) * sizeof(std::uint32_t) >
          ggml_cuda_info().devices[launch.device()].smpbo) {
    return Rejected(
        "D2R requires bounded raw Q2_K expert slots, contiguous output and supported MMA");
  }
  const auto j = ggml_cuda_mmq_get_J_max(w->type, w->ne[1] % 128 != 0, cc, 128);
  const auto padded = GGML_PAD(x->ne[0], MATRIX_ROW_PADDING);
  Draws draws;
  draws.Add(static_cast<std::uint64_t>(rows) * sizeof(std::int32_t));
  draws.Add(static_cast<std::uint64_t>(rows + j) * sizeof(std::int32_t));
  draws.Add(static_cast<std::uint64_t>(w->ne[2] + 1) * sizeof(std::int32_t));
  draws.Add(static_cast<std::uint64_t>(rows * padded) * sizeof(block_q8_1_mmq) / QK8_1_MMQ +
            static_cast<std::uint64_t>(j) * sizeof(block_q8_1_mmq));
  draws.Add(ds4_mmq_q2_K_moe_d2r_scratch_bytes(rows, static_cast<int>(w->ne[2])));
  return draws.bytes;
}

std::expected<void, KernelFailure> MulMatIdQ2D2r(LaunchContext& launch, ggml_tensor* node) {
  auto bytes = PlanMulMatIdQ2D2r(launch, node);
  if (!bytes) return std::unexpected(bytes.error());
  return launch.Run(base::Bytes(*bytes), [node](ggml_backend_cuda_context& c) {
    const auto* w = node->src[0];
    const auto* x = node->src[1];
    const auto* ids = node->src[2];
    const auto used = ids->ne[0];
    const auto tokens = x->ne[2];
    const auto rows = used * tokens;
    const auto padded = GGML_PAD(x->ne[0], MATRIX_ROW_PADDING);
    const auto j = ggml_cuda_mmq_get_J_max(w->type, w->ne[1] % 128 != 0,
                                           ggml_cuda_info().devices[c.device].cc, 128);
    const auto stream = c.stream();
    ggml_cuda_pool_alloc<std::int32_t> src(c.pool(), static_cast<std::size_t>(rows));
    ggml_cuda_pool_alloc<std::int32_t> dst(c.pool(), static_cast<std::size_t>(rows + j));
    ggml_cuda_pool_alloc<std::int32_t> bounds(c.pool(), static_cast<std::size_t>(w->ne[2] + 1));
    ggml_cuda_launch_mm_ids_helper(
        static_cast<const std::int32_t*>(ids->data), src.get(), dst.get(), bounds.get(),
        static_cast<int>(w->ne[2]), static_cast<int>(tokens), static_cast<int>(used),
        static_cast<int>(x->ne[1]), static_cast<int>(ids->nb[1] / sizeof(std::int32_t)),
        static_cast<int>(x->nb[2] / x->nb[1]), false, stream);
    CUDA_CHECK(cudaGetLastError());
    if (internal::CudaErrorPending()) return;
    const auto quant_bytes =
        static_cast<std::size_t>(rows * padded) * sizeof(block_q8_1_mmq) / QK8_1_MMQ +
        static_cast<std::size_t>(j) * sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> quant(c.pool(), quant_bytes);
    quantize_mmq_q8_1_cuda(static_cast<const float*>(x->data), src.get(), quant.get(), w->type,
                           x->ne[0], static_cast<std::int64_t>(x->nb[1] / sizeof(float)),
                           static_cast<std::int64_t>(x->nb[2] / sizeof(float)),
                           static_cast<std::int64_t>(x->nb[3] / sizeof(float)), padded, rows, 1, 1,
                           stream);
    CUDA_CHECK(cudaGetLastError());
    if (internal::CudaErrorPending()) return;
    const auto work_bytes = ds4_mmq_q2_K_moe_d2r_scratch_bytes(rows, static_cast<int>(w->ne[2]));
    ggml_cuda_pool_alloc<char> work(c.pool(), work_bytes);
    const int result = llmp_q2_K_raw_d2r_launch(
        w->data, ggml_nbytes(w), w->nb[1], w->nb[2], quant.get(), dst.get(), bounds.get(),
        static_cast<float*>(node->data), static_cast<int>(w->ne[1]), static_cast<int>(w->ne[0]),
        rows, static_cast<int>(w->ne[2]), work.get(), work_bytes, stream);
    // Every allocation is inside this planned Run scope. Graph captures pin
    // the same cataloged workspace; no pool pointer survives across calls.
    // The upstream launcher consumes cudaGetLastError. Record every refusal
    // here so LaunchContext faults and retains workspace on uncertain work.
    if (result != 0)
      ggml_cuda_error("D2R launch", __func__, __FILE__, __LINE__,
                      "D2R launcher failed after preparation");
  });
}
}  // namespace llmp::kernels::ggml
