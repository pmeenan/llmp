// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>

#include "base/bytes.h"
#include "base/check.h"
#include "common.cuh"
#include "cpy.cuh"
#include "im2col.cuh"
#include "kernels/ggml/conv2d_tiled.h"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/ggml_support.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/mul_mat_cublas_internal.h"
#include "kernels/ggml/validate.h"
#include "kernels/ggml/validate_util.h"

namespace llmp::kernels::ggml {
namespace {
using detail::Rejected;

// These are private stack descriptors, not new IR nodes or retained allocations.
// Parameters and source pointers are consumed by the host launchers during Run;
// captured kernels retain only the paid device addresses and scalar dimensions.
struct Tile {
  ggml_tensor input{}, columns{}, weights{}, matrix{}, product{}, transpose{}, packed{}, output{};
};
void Packed(ggml_tensor& t, std::array<std::int64_t, 4> dims, void* data) {
  t = {};
  t.type = GGML_TYPE_F32;
  t.data = data;
  std::size_t stride = sizeof(float);
  for (std::size_t i = 0; i < dims.size(); ++i) {
    t.ne[i] = dims[i];
    t.nb[i] = stride;
    stride *= static_cast<std::size_t>(dims[i]);
  }
}
void MakeTile(Tile& t, const ggml_tensor* node, const Conv2dTilePlan& plan, std::int64_t batch,
              std::int64_t y, std::int64_t rows, std::byte* scratch) {
  const auto width = node->ne[0];
  const auto count = width * rows;
  const auto channels = node->ne[2];
  const auto k = 9 * node->src[0]->ne[2];
  t.input = *node->src[1];
  t.input.ne[3] = 1;
  t.input.data =
      static_cast<std::byte*>(t.input.data) + static_cast<std::size_t>(batch) * t.input.nb[3];
  t.input.view_src = nullptr;
  t.input.view_offs = 0;
  Packed(t.columns, {k, width, rows, 1}, scratch);
  t.columns.op = GGML_OP_IM2COL;
  t.columns.src[0] = node->src[0];
  t.columns.src[1] = &t.input;
  t.columns.op_params[0] = t.columns.op_params[1] = 1;
  t.columns.op_params[2] = 1;
  // Original IH and channel pitch remain intact. Only the requested output
  // stripe is shifted; the upstream halo checks use original input bounds.
  t.columns.op_params[3] = static_cast<std::int32_t>(1 - y);
  t.columns.op_params[4] = t.columns.op_params[5] = t.columns.op_params[6] = 1;
  Packed(t.weights, {k, channels, 1, 1}, node->src[0]->data);
  Packed(t.matrix, {k, count, 1, 1}, scratch);
  Packed(t.product, {channels, count, 1, 1}, scratch + plan.columns_bytes);
  t.product.op = GGML_OP_MUL_MAT;
  t.product.src[0] = &t.weights;
  t.product.src[1] = &t.matrix;
  Packed(t.transpose, {count, channels, 1, 1}, t.product.data);
  t.transpose.nb[0] = static_cast<std::size_t>(channels) * sizeof(float);
  t.transpose.nb[1] = sizeof(float);
  Packed(t.packed, {count, channels, 1, 1}, scratch + plan.columns_bytes + plan.product_bytes);
  t.packed.op = GGML_OP_CONT;
  t.packed.src[0] = &t.transpose;
  auto* destination = static_cast<std::byte*>(node->data) +
                      static_cast<std::size_t>(batch) * node->nb[3] +
                      static_cast<std::size_t>(y) * node->nb[1];
  Packed(t.output, {count, channels, 1, 1}, destination);
  // Same-shape source/destination, dimension0 contiguous, dimension1 carries
  // rows, higher dimensions1: exactly pinned cpy's cudaMemcpy2DAsync path.
  t.output.nb[1] = node->nb[2];
  t.output.nb[2] = t.output.nb[3] = node->nb[3];
}
std::expected<CublasMulMat, KernelFailure> CheckTile(Tile& t) {
  // The selected compound implementation explicitly uses the existing F32
  // cuBLAS body. Ordinary MulMat's upstream selector and checks are unchanged.
  auto product = CheckMulMatCublas(&t.product, GGML_TYPE_F32, true);
  if (!product) return std::unexpected(product.error());
  if (product->scratch != 0 || product->weights != CublasOperand::kDirect ||
      product->input != CublasOperand::kDirect)
    return Rejected("packed F32 convolution product unexpectedly requires conversion scratch");
  if (auto copy = CheckCont(&t.packed); !copy) return std::unexpected(copy.error());
  if (!detail::AllSane({&t.output}) || !detail::Disjoint(&t.output, &t.packed, false) ||
      t.output.nb[1] < static_cast<std::size_t>(t.output.ne[0]) * sizeof(float))
    return Rejected("convolution stripe is not a disjoint pitched destination");
  return product;
}
}  // namespace

std::expected<std::uint64_t, KernelFailure> PlanConv2dTiledF32(const LaunchContext&,
                                                               const ggml_tensor* node) {
  const auto plan = CheckConv2dTiledF32(node);
  if (!plan) return std::unexpected(plan.error());
  // Descriptor-only planning: these disjoint addresses are never read. Actual
  // binding repeats the same checks with the lent pool and source addresses.
  auto original = *node;
  auto weight = *node->src[0];
  auto input = *node->src[1];
  original.data = reinterpret_cast<void*>(0x100000000ULL);
  weight.data = reinterpret_cast<void*>(0x200000000ULL);
  input.data = reinterpret_cast<void*>(0x300000000ULL);
  original.src[0] = &weight;
  original.src[1] = &input;
  const auto rows = static_cast<std::int64_t>(plan->rows);
  const auto tail = (node->ne[1] - 1) % rows + 1;
  for (const auto h : {rows, tail}) {
    Tile tile;
    MakeTile(tile, &original, *plan, 0, node->ne[1] - h, h,
             reinterpret_cast<std::byte*>(0x400000000ULL));
    if (auto checked = CheckTile(tile); !checked) return std::unexpected(checked.error());
  }
  return plan->scratch;
}

std::expected<void, KernelFailure> Conv2dTiledF32(LaunchContext& launch, ggml_tensor* node) {
  auto plan = CheckConv2dTiledF32(node);
  if (!plan) return std::unexpected(plan.error());
  if (launch.cublas() == nullptr)
    return Rejected("tiled convolution requires the lent cuBLAS handle");
  const auto workspace = launch.workspace();
  if (workspace.size.value() < plan->scratch)
    return Rejected("tiled convolution exceeds the lent workspace");
  for (const auto space : {workspace, launch.cublas()->workspace()}) {
    if (auto clear = CheckClearOf(node, space.base, space.size.value()); !clear) return clear;
  }
  // The provider also protects the library workspace from overlap with the
  // launch pool. Check explicitly before creating any compound operand.
  const auto library = launch.cublas()->workspace();
  if (library.size.value() != 0 && workspace.base < library.base + library.size.value() &&
      library.base < workspace.base + workspace.size.value())
    return Rejected("convolution scratch overlaps cuBLAS workspace");
  const auto height = node->ne[1];
  const auto rows = static_cast<std::int64_t>(plan->rows);
  const auto tail = (height - 1) % rows + 1;
  std::array<CublasMulMat, 2> products{};
  // All refusals precede Run. Two shapes suffice: maximum stripe and tail;
  // output offsets/batch differences stay inside the checked original span.
  for (std::size_t i = 0; i < products.size(); ++i) {
    Tile tile;
    MakeTile(tile, node, *plan, i == 0 ? 0 : node->ne[3] - 1, i == 0 ? 0 : height - tail,
             i == 0 ? rows : tail, reinterpret_cast<std::byte*>(workspace.base));
    auto checked = CheckTile(tile);
    if (!checked) return std::unexpected(checked.error());
    products[i] = *checked;
  }
  return launch.Run(base::Bytes(plan->scratch), [&](ggml_backend_cuda_context& context) {
    ggml_cuda_pool_alloc<std::byte> scratch(context.pool(),
                                            static_cast<std::size_t>(plan->scratch));
    base::Check(reinterpret_cast<std::uintptr_t>(scratch.get()) == workspace.base,
                "a compound convolution starts at its checked pool base");
    for (std::int64_t batch = 0; batch < node->ne[3]; ++batch) {
      for (std::int64_t y = 0; y < height; y += rows) {
        Tile tile;
        const auto h = std::min(rows, height - y);
        MakeTile(tile, node, *plan, batch, y, h, scratch.get());
        ggml_cuda_op_im2col(context, &tile.columns);
        if (internal::CudaErrorPending()) return;
        internal::LaunchPlannedMulMatCublas(context, products[h == rows ? 0 : 1], &tile.product);
        if (internal::CudaErrorPending()) return;
        ggml_cuda_cpy(context, &tile.transpose, &tile.packed);
        if (internal::CudaErrorPending()) return;
        ggml_cuda_cpy(context, &tile.packed, &tile.output);
        if (internal::CudaErrorPending()) return;
      }
    }
  });
}
}  // namespace llmp::kernels::ggml
