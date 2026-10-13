// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include <array>
#include <cstring>
#include <limits>

#include "base/bytes.h"
#include "base/check.h"
#include "common.cuh"
#include "cpy.cuh"
#include "fill.cuh"
#include "kernels/ggml/attention_f32.h"
#include "kernels/ggml/fattn_mma.h"
#include "kernels/ggml/ggml_support.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/ops_ext.h"
#include "kernels/ggml/validate.h"
#include "kernels/ggml/validate_ext.h"
#include "kernels/ggml/validate_util.h"
namespace llmp::kernels::ggml {
namespace {
struct PackedAttention {
  std::array<ggml_tensor, 3> padded{}, real{};
  ggml_tensor output{};
};
void Packed(ggml_tensor& t, std::array<std::int64_t, 4> dims, void* data, float fill) {
  t = {};
  t.type = GGML_TYPE_F16;
  t.op = GGML_OP_FILL;
  t.data = data;
  std::memcpy(t.op_params, &fill, sizeof(fill));
  std::size_t stride = sizeof(ggml_fp16_t);
  for (std::size_t i = 0; i < dims.size(); ++i) {
    t.ne[i] = dims[i];
    t.nb[i] = stride;
    stride *= static_cast<std::size_t>(dims[i]);
  }
}
void Make(PackedAttention& a, const ggml_tensor* node, const AttentionF32Plan& p,
          std::byte* scratch) {
  const auto cells = static_cast<std::int64_t>(p.cells);
  const auto rows = static_cast<std::int64_t>(p.rows);
  Packed(a.padded[0], {128, cells, node->src[1]->ne[2], 1}, scratch, 0);
  Packed(a.padded[1], {128, cells, node->src[2]->ne[2], 1}, scratch + p.kv_bytes, 0);
  Packed(a.padded[2], {cells, rows, 1, 1}, scratch + 2 * p.kv_bytes,
         -std::numeric_limits<float>::infinity());
  for (std::size_t i = 0; i < a.real.size(); ++i) {
    a.real[i] = a.padded[i];
    // Same number/order of real elements as the source; padded row/head
    // pitches stay intact. Existing scalar cpy converts to those addresses.
    for (int d = 0; d < GGML_MAX_DIMS; ++d) a.real[i].ne[d] = node->src[i + 1]->ne[d];
  }
  a.output = *node;
  for (std::size_t i = 0; i < a.padded.size(); ++i) a.output.src[i + 1] = &a.padded[i];
}
std::expected<FlashAttnMmaPlan, KernelFailure> Physical(const LaunchContext& launch,
                                                        const ggml_tensor* node,
                                                        const AttentionF32Plan& plan,
                                                        std::byte* scratch) {
  PackedAttention packed;
  Make(packed, node, plan, scratch);
  for (const auto& t : packed.padded)
    if (auto check = CheckFill(&t); !check) return std::unexpected(check.error());
  return PlanFlashAttnMma128(launch, &packed.output);
}
}  // namespace
std::expected<std::uint64_t, KernelFailure> PlanAttentionF32(const LaunchContext& launch,
                                                             const ggml_tensor* node) {
  auto plan = CheckAttentionF32(node);
  if (!plan) return std::unexpected(plan.error());
  auto output = *node;
  auto query = *node->src[0];
  output.data = reinterpret_cast<void*>(0x100000000ULL);
  output.view_src = nullptr;
  output.view_offs = 0;
  query.data = reinterpret_cast<void*>(0x200000000ULL);
  query.view_src = nullptr;
  query.view_offs = 0;
  output.src[0] = &query;
  auto flash = Physical(launch, &output, *plan, reinterpret_cast<std::byte*>(0x1000000000ULL));
  if (!flash) return std::unexpected(flash.error());
  return plan->packing_bytes + flash->scratch;
}
std::expected<void, KernelFailure> AttentionF32(LaunchContext& launch, ggml_tensor* node) {
  auto plan = CheckAttentionF32(node);
  if (!plan) return std::unexpected(plan.error());
  const auto workspace = launch.workspace();
  auto* scratch = reinterpret_cast<std::byte*>(workspace.base);
  auto flash = Physical(launch, node, *plan, scratch);
  if (!flash) return std::unexpected(flash.error());
  const auto bytes = plan->packing_bytes + flash->scratch;
  if (bytes > workspace.size.value()) return detail::Rejected("attention exceeds lent scratch");
  if (auto clear = CheckClearOf(node, workspace.base, bytes); !clear) return clear;
  const auto run = detail::FlashAttnMmaCase128(flash->columns);
  if (run == nullptr) return detail::Rejected("attention has no compiled query tile");
  return launch.Run(base::Bytes(bytes), [&](ggml_backend_cuda_context& context) {
    ggml_cuda_pool_alloc<std::byte> memory(context.pool(), plan->packing_bytes);
    base::Check(reinterpret_cast<std::uintptr_t>(memory.get()) == workspace.base,
                "attention packing starts at the paid pool base");
    PackedAttention packed;
    Make(packed, node, *plan, memory.get());
    for (std::size_t i = 0; i < packed.padded.size(); ++i) {
      ggml_cuda_op_fill(context, &packed.padded[i]);
      if (internal::CudaErrorPending()) return;
      ggml_cuda_cpy(context, node->src[i + 1], &packed.real[i]);
      if (internal::CudaErrorPending()) return;
    }
    run(context, &packed.output);
  });
}
}  // namespace llmp::kernels::ggml
