// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// Exact d81235049384534c167caea52b85a694f6103d14 mmvq.cu dense preparation
// and prequantized-consumer calls. Only the duplicate preparation is removed.
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>

#include "base/bytes.h"
#include "common.cuh"
#include "gemma31_mmvq_pair.h"
#include "kernels/ggml/ops_ext.h"
#include "kernels/ggml/validate_util.h"
#include "mmvq.cuh"
#include "quantize.cuh"

namespace jitllm::benchmarks::gemma31_mmvq {
namespace kg = kernels::ggml;
namespace {
std::unexpected<kg::KernelFailure> Refuse(std::string message) {
  return std::unexpected(
      kg::KernelFailure{.error = kg::KernelError::kRejected, .detail = std::move(message)});
}
bool Shape(const ggml_tensor* t, ggml_type type, std::int64_t width, std::int64_t columns) {
  return t != nullptr && t->type == type && t->ne[0] == width && t->ne[1] == columns &&
         t->ne[2] == 1 && t->ne[3] == 1 && t->view_src == nullptr && kg::detail::Packed(t);
}
constexpr std::int64_t kWidth = 5376, kFfn = 21504, kColumns = 2;
constexpr std::int64_t kPadded = 5632;
static_assert(MATRIX_ROW_PADDING == 512 && QK8_1 == 32);
static_assert(sizeof(block_q8_1) == 36);
constexpr std::uint64_t kScratch = kColumns * kPadded * sizeof(block_q8_1) / QK8_1;
static_assert(kScratch == 12672);
// Include the original MMVQ readable weight tail in every cross-alias check.
// The individual primitive plan requires the binder's authenticated marker.
bool Separate(const ggml_tensor* a, std::uint64_t a_tail, const ggml_tensor* b,
              std::uint64_t b_tail) {
  auto na = kg::detail::Extent(a), nb = kg::detail::Extent(b);
  if (!na || !nb || *na > UINT64_MAX - a_tail || *nb > UINT64_MAX - b_tail) return false;
  const auto aa = reinterpret_cast<std::uintptr_t>(a->data);
  const auto ab = reinterpret_cast<std::uintptr_t>(b->data);
  *na += a_tail;
  *nb += b_tail;
  return aa <= UINT64_MAX - *na && ab <= UINT64_MAX - *nb && (aa + *na <= ab || ab + *nb <= aa);
}
}  // namespace

std::expected<std::uint64_t, kg::KernelFailure> PlanOrdinaryC2Pair(const kg::LaunchContext& launch,
                                                                   const ggml_tensor* gate,
                                                                   const ggml_tensor* up) {
  if (gate == nullptr || up == nullptr || gate == up || gate->op != GGML_OP_MUL_MAT ||
      up->op != GGML_OP_MUL_MAT || gate->src[1] == nullptr || gate->src[1] != up->src[1] ||
      !Shape(gate->src[1], GGML_TYPE_F32, kWidth, kColumns) ||
      !Shape(gate->src[0], GGML_TYPE_Q4_K, kWidth, kFfn) ||
      !Shape(up->src[0], GGML_TYPE_Q4_K, kWidth, kFfn) || gate->src[0] == up->src[0] ||
      gate->src[0]->op != GGML_OP_NONE || up->src[0]->op != GGML_OP_NONE ||
      !Shape(gate, GGML_TYPE_F32, kFfn, kColumns) || !Shape(up, GGML_TYPE_F32, kFfn, kColumns))
    return Refuse("private ordinary MMVQ pair requires exact dense31 C2 Q4_K operands");
  for (int i = 2; i < GGML_MAX_SRC; ++i)
    if (gate->src[i] != nullptr || up->src[i] != nullptr)
      return Refuse("private ordinary MMVQ pair rejects additional source operands");
  if (ggml_cuda_info().devices[launch.device()].cc != 1210)
    return Refuse("private ordinary MMVQ pair is scoped to GB10");
  // Both complete original plans are required BEFORE a pool draw or enqueue.
  auto a = kg::PlanMulMatVecQ(launch, gate), b = kg::PlanMulMatVecQ(launch, up);
  if (!a) return std::unexpected(a.error());
  if (!b) return std::unexpected(b.error());
  if (*a != kScratch || *b != kScratch)
    return Refuse("original ordinary MMVQ plans disagree with the shared preparation size");
  const std::uint64_t tail = ggml_row_size(GGML_TYPE_Q4_K, kPadded - kWidth);
  const std::array<const ggml_tensor*, 5> all{gate->src[0], up->src[0], gate->src[1], gate, up};
  for (std::size_t i = 0; i < all.size(); ++i)
    for (std::size_t j = i + 1; j < all.size(); ++j)
      if (!Separate(all[i], i < 2 ? tail : 0, all[j], j < 2 ? tail : 0))
        return Refuse("private ordinary MMVQ pair aliases operands or readable weight tails");
  return kScratch;
}

std::expected<void, kg::KernelFailure> OrdinaryC2Pair(kg::LaunchContext& launch, ggml_tensor* gate,
                                                      ggml_tensor* up) {
  auto plan = PlanOrdinaryC2Pair(launch, gate, up);
  if (!plan) return std::unexpected(plan.error());
  return launch.Run(base::Bytes(*plan), [gate, up](ggml_backend_cuda_context& context) {
    const auto* x = gate->src[1];
    ggml_cuda_pool_alloc<char> q8(context.pool(), kScratch);
    const auto stream = context.stream();
    quantize_row_q8_1_cuda(static_cast<const float*>(x->data), nullptr, q8.get(), GGML_TYPE_Q4_K,
                           kWidth, static_cast<std::int64_t>(x->nb[1] / sizeof(float)),
                           static_cast<std::int64_t>(x->nb[2] / sizeof(float)),
                           static_cast<std::int64_t>(x->nb[3] / sizeof(float)), kPadded, kColumns,
                           1, 1, stream);
    CUDA_CHECK(cudaGetLastError());
    for (auto* product : {gate, up})
      ggml_cuda_op_mul_mat_vec_q(
          context, product->src[0], x, product, static_cast<const char*>(product->src[0]->data),
          static_cast<const float*>(x->data), q8.get(), static_cast<float*>(product->data), 0, kFfn,
          kColumns, kPadded, stream);
  });
}
}  // namespace jitllm::benchmarks::gemma31_mmvq
