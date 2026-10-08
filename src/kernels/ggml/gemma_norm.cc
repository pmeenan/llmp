// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/gemma_norm.h"

#include <cmath>
#include <limits>

#include "kernels/ggml/validate.h"
#include "kernels/ggml/validate_util.h"

namespace llmp::kernels::ggml {
namespace {
using namespace detail;
const ggml_tensor* Weight(const ggml_tensor* norm, const ggml_tensor* mul) {
  return mul->src[0] == norm ? mul->src[1] : mul->src[0];
}
bool Chain(const ggml_tensor* tensor) {
  for (unsigned i = 0; tensor != nullptr && i < 64; ++i) {
    if (tensor->view_src == nullptr) return true;
    if (tensor->view_offs > std::numeric_limits<std::uint64_t>::max() -
                                reinterpret_cast<std::uintptr_t>(tensor->view_src->data))
      return false;
    tensor = tensor->view_src;
  }
  return tensor == nullptr;
}
bool OperandChains(const ggml_tensor* tensor) {
  if (!Chain(tensor)) return false;
  if (tensor != nullptr)
    for (const auto* source : tensor->src)
      if (!Chain(source)) return false;
  return true;
}
bool Elided(const ggml_tensor* tensor, const ggml_tensor* norm, const ggml_tensor* mul) {
  return tensor != nullptr && (Root(tensor) == norm || Root(tensor) == mul);
}
std::expected<void, KernelFailure> Norm(const ggml_tensor* norm, const ggml_tensor* mul) {
  if (!OperandChains(norm) || !OperandChains(mul) ||
      (norm != nullptr && norm->view_src != nullptr) ||
      (mul != nullptr && mul->view_src != nullptr))
    return Rejected("norm/product intermediates must be nonviews with bounded operand chains");
  if (auto r = CheckRmsNormMul(norm, mul); !r) return r;
  if (Elided(norm->src[0], norm, mul) || Elided(Weight(norm, mul), norm, mul))
    return Rejected("actual norm input depends on an elided intermediate");
  if (!std::isfinite(ParamF32(norm, 0))) return Rejected("norm epsilon must be finite");
  return {};
}
}  // namespace

std::expected<void, KernelFailure> CheckGemmaNormRope(const ggml_tensor* norm,
                                                      const ggml_tensor* mul,
                                                      const ggml_tensor* rope) {
  if (auto r = Norm(norm, mul); !r) return r;
  if (!OperandChains(rope)) return Rejected("invalid RoPE operand view chain");
  if (rope == nullptr || rope->op != GGML_OP_ROPE || !IsF32(rope) || rope->src[0] != mul ||
      (mul->ne[0] != 256 && mul->ne[0] != 512) || rope->op_params[1] != mul->ne[0] ||
      rope->op_params[4] <= 0 || !Extent(rope) || !Packed(mul) || !Packed(rope))
    return Rejected("norm/RoPE requires full D256/D512 rotation and packed tensors");
  if (auto r = CheckRope(rope); !r) return r;
  if (Elided(rope->src[1], norm, mul) || Elided(rope->src[2], norm, mul))
    return Rejected("RoPE metadata depends on an elided intermediate");
  for (int p = 5; p <= 10; ++p)
    if (!std::isfinite(ParamF32(rope, p))) return Rejected("nonfinite RoPE parameter");
  if (ParamF32(rope, 5) <= 1.0f || ParamF32(rope, 6) <= 0.0f || ParamF32(rope, 8) <= 0.0f ||
      ParamF32(rope, 9) <= 0.0f || ParamF32(rope, 10) <= 0.0f)
    return Rejected("invalid RoPE frequency/correction parameters");
  const auto* x = norm->src[0];
  const auto* w = Weight(norm, mul);
  // The fused launch reads the pre-normalization source, not mul's bytes.
  // Exact in-place rotation would also destroy raw K used independently as V.
  if (!FitsRowGrid(x) || !Disjoint(rope, x, false) || !Disjoint(rope, w, false))
    return Rejected("norm/RoPE destination overlaps a source, or exceeds its row grid");
  return {};
}

std::expected<void, KernelFailure> CheckGemmaNormAdd(const ggml_tensor* norm,
                                                     const ggml_tensor* mul,
                                                     const ggml_tensor* add) {
  if (auto r = Norm(norm, mul); !r) return r;
  if (!OperandChains(add)) return Rejected("invalid residual operand view chain");
  if (add == nullptr || (add->src[0] != mul && add->src[1] != mul) || !IsF32(add))
    return Rejected("norm/residual ADD must consume this product");
  if (norm->ne[0] != 2304 && norm->ne[0] != 2560 && norm->ne[0] != 2816 && norm->ne[0] != 5376)
    return Rejected("norm/residual is bounded to approved Gemma widths");
  if (auto r = CheckBinary(add, GGML_OP_ADD); !r) return r;
  const auto* residual = add->src[0] == mul ? add->src[1] : add->src[0];
  if (Elided(residual, norm, mul)) return Rejected("residual depends on an elided intermediate");
  if (!IsF32(residual) || !ggml_are_same_shape(add, norm) || !ggml_is_contiguous(add->src[0]) ||
      !ggml_is_contiguous_rows(add->src[1]))
    return Rejected("norm/residual needs contiguous F32 operands of the norm's shape");
  if (!Disjoint(add, norm->src[0], false) || !Disjoint(add, Weight(norm, mul), false) ||
      !Disjoint(add, residual, false))
    return Rejected("norm/residual destination overlaps an actual source");
  return {};
}
std::expected<void, KernelFailure> CheckGemmaNormAddGather(const ggml_tensor* norm,
                                                           const ggml_tensor* mul,
                                                           const ggml_tensor* gather,
                                                           const ggml_tensor* add) {
  if (auto checked = CheckGemmaNormAdd(norm, mul, add); !checked) return checked;
  const auto* residual = add->src[0] == mul ? add->src[1] : add->src[0];
  if (gather == nullptr || gather != residual || gather->op != GGML_OP_GET_ROWS ||
      gather->view_src != nullptr || !OperandChains(gather))
    return Rejected("norm/add preparation must be this materialized residual GET_ROWS");
  if (Elided(gather->src[0], norm, mul) || Elided(gather->src[1], norm, mul))
    return Rejected("residual GET_ROWS reads an elided intermediate");
  if (auto checked = CheckGetRows(gather); !checked) return checked;
  if (!Disjoint(gather, norm->src[0], false) || !Disjoint(gather, Weight(norm, mul), false))
    return Rejected("early residual GET_ROWS overwrites an actual norm input");
  return {};
}
}  // namespace llmp::kernels::ggml
