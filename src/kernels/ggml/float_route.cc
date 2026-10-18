// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// NVIDIA branches of ggml_cuda_should_use_mmf/mmvf at locked
// d81235049384534c167caea52b85a694f6103d14 (mmf.cu:133, mmvf.cu:793).
// No device launches or CUDA headers: selection and its expansion gate can
// inspect precisely the same shape/stride domain in every build profile.
#include "kernels/ggml/float_route.h"

#include <algorithm>
#include <array>
#include <cstdint>

#include "kernels/ggml/capabilities.h"

namespace llmp::kernels::ggml {
namespace {
using format::Encoding;
const execution::OperandFormat* Operand(const execution::CapabilityQuery& q, std::uint8_t index) {
  const auto at = std::ranges::find(q.operands, index, &execution::OperandFormat::operand);
  return at == q.operands.end() ? nullptr : &*at;
}
bool AlignedRows(const execution::OperandFormat& w, std::uint64_t size) {
  if (!w.memory || w.memory->nb[0] != size) return false;
  for (std::size_t d = 1; d < 4; ++d)
    if (w.memory->nb[d] % (2 * size)) return false;
  return true;
}
bool Mmf(const execution::OperandFormat& w, const execution::CapabilityQuery& q) {
  const std::uint64_t size = w.encoding == Encoding::kF32 ? 4 : 2;
  if (w.ne[0] % (32 * (4 / size)) || !AlignedRows(w, size) || w.ne[1] % 32 || q.rows > 16)
    return false;
  // The supported NVIDIA builds contain sm_86 and/or sm_121[a] images;
  // both have the pinned helper's Ampere MMA branch. CompiledFor below
  // prevents treating an image for another architecture as evidence.
  return q.arch >= 80 && CompiledFor(q.arch);
}
bool Mmvf(const execution::OperandFormat& w, const execution::CapabilityQuery& q) {
  const std::uint64_t size = w.encoding == Encoding::kF32 ? 4 : 2;
  if (w.ne[0] % 2 || !AlignedRows(w, size)) return false;
  if (w.encoding == Encoding::kF32) return q.rows <= 3;
  if (w.encoding != Encoding::kF16 && w.encoding != Encoding::kBF16) return false;
  const bool small = w.ne[1] <= 512 || w.ne[2] * w.ne[3] == 1;
  if (small && !Mmf(w, q)) return q.rows <= 8;  // pinned MMVF_MAX_BATCH_SIZE
  return small && q.rows == 1;                  // Ampere MMA on the supported images
}
}  // namespace
bool FloatProductUsesCublas(const execution::CapabilityQuery& q) {
  const auto* w = Operand(q, 0);
  const auto* x = Operand(q, 1);
  const auto* y = Operand(q, execution::kOutputOperand);
  if (!w || !x || !w->ne[0] || !w->memory || !CompiledFor(q.arch)) return true;
  if (x->encoding != Encoding::kF32 || (y && y->encoding != Encoding::kF32)) return true;
  return !Mmvf(*w, q) && !Mmf(*w, q);
}
}  // namespace llmp::kernels::ggml
