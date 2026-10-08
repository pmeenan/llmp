// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/dsv4_weighted_reduce.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>

#include "kernels/ggml/llmp_ops.h"
#include "kernels/ggml/validate_util.h"

namespace llmp::kernels::ggml {
namespace {

bool Shape(const ggml_tensor* tensor, const std::array<std::int64_t, 4>& shape) {
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (tensor->ne[i] != shape[i]) {
      return false;
    }
  }
  return true;
}

bool ContiguousStrides(const ggml_tensor* tensor) {
  std::uint64_t stride = sizeof(float);
  for (int i = 0; i < GGML_MAX_DIMS; ++i) {
    if (tensor->nb[i] != stride) {
      return false;
    }
    // Called only after the fixed shapes have been checked. The largest
    // complete operand is 4096*6*4096 F32, so this cannot overflow.
    stride *= static_cast<std::uint64_t>(tensor->ne[i]);
  }
  return true;
}

struct Access {
  std::uint64_t address = 0;
  std::uint64_t bytes = 0;
};

bool Overlap(const Access& a, const Access& b) {
  // Extent and the full supplied range were checked before either sum.
  return a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}

}  // namespace

bool Dsv4WeightedReduceFits(const ggml_tensor* down, const ggml_tensor* weights) {
  if (!detail::IsF32(down) || !detail::IsF32(weights)) {
    return false;
  }
  const std::int64_t rows = down->ne[2];
  return rows > 0 && rows <= kDsv4WeightedReduceMaxRows &&
         Shape(down, {kDsv4WeightedReduceWidth, kDsv4WeightedReduceSlots, rows, 1}) &&
         Shape(weights, {1, kDsv4WeightedReduceSlots, rows, 1}) && ContiguousStrides(down) &&
         ContiguousStrides(weights);
}

std::expected<void, KernelFailure> CheckDsv4WeightedReduce(const Dsv4WeightedReduce& desc) {
  const std::array operands = {desc.down, desc.weights, desc.values};
  for (const auto& operand : operands) {
    if (!detail::IsF32(operand.tensor) || !detail::Bound(operand.tensor)) {
      return detail::Rejected("dsv4 weighted reduction requires three bound F32 tensors");
    }
  }
  const std::int64_t rows = desc.down.tensor->ne[2];
  if (!Dsv4WeightedReduceFits(desc.down.tensor, desc.weights.tensor) ||
      !Shape(desc.values.tensor, {kDsv4WeightedReduceWidth, rows, 1, 1})) {
    return detail::Rejected(
        "dsv4 weighted reduction requires width4096, six slots and rows1..4096");
  }
  std::array<Access, 3> accessed{};
  for (std::size_t i = 0; i < operands.size(); ++i) {
    const auto& operand = operands[i];
    if (!ContiguousStrides(operand.tensor) || !detail::Aligned(operand.tensor, alignof(float))) {
      return detail::Rejected("dsv4 weighted reduction requires contiguous, aligned F32 operands");
    }
    const auto address = reinterpret_cast<std::uintptr_t>(operand.tensor->data);
    const auto extent = detail::Extent(operand.tensor);
    if (!extent || operand.bytes.value() < *extent ||
        operand.bytes.value() > std::numeric_limits<std::uint64_t>::max() - address) {
      return detail::Rejected("dsv4 weighted reduction operand is short or has an invalid end");
    }
    accessed[i] = {.address = address, .bytes = *extent};
  }
  if (Overlap(accessed[2], accessed[0]) || Overlap(accessed[2], accessed[1])) {
    return detail::Rejected("dsv4 weighted reduction output overlaps an input");
  }
  return {};
}

std::expected<void, KernelFailure> CheckDsv4OrderedReduce(const ggml_tensor* node) {
  if (LlmpOpOf(node) != LlmpOp::kDsv4WeightedReduce || !detail::Bound(node) ||
      !detail::Bound(node->src[0]) || !detail::Bound(node->src[1])) {
    return detail::Rejected("not a bound ordered weighted-reduction node");
  }
  for (int i = 2; i < GGML_MAX_SRC; ++i) {
    if (node->src[i] != nullptr) {
      return detail::Rejected("ordered weighted reduction takes exactly two operands");
    }
  }
  if (!Dsv4WeightedReduceFits(node->src[0], node->src[1]) || !detail::IsF32(node) ||
      !Shape(node, {kDsv4WeightedReduceWidth, node->src[0]->ne[2], 1, 1}) ||
      !ContiguousStrides(node)) {
    return detail::Rejected("ordered weighted-reduction graph metadata differs");
  }
  // The executor protects the graph's bound allocation extents through
  // completion. These exact shapes also bound ggml_nbytes arithmetic.
  return CheckDsv4WeightedReduce({.down = {node->src[0], base::Bytes(ggml_nbytes(node->src[0]))},
                                  .weights = {node->src[1], base::Bytes(ggml_nbytes(node->src[1]))},
                                  .values = {node, base::Bytes(ggml_nbytes(node))}});
}

}  // namespace llmp::kernels::ggml
