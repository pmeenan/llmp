// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// GGML binary operations over one-element rows (ne[0] == 1, packed ne[1]),
// shifted left so the broadcast kernels see whole rows: ops.cu's and
// ops_ext.cu's binary launches.

#ifndef LLMP_KERNELS_GGML_BINARY_SHIFT_H_
#define LLMP_KERNELS_GGML_BINARY_SHIFT_H_

#include <cstddef>

#include "ggml.h"

namespace llmp::kernels::ggml {

inline bool CanShiftLeft(const ggml_tensor& t) {
  const std::size_t elem = ggml_type_size(t.type);
  return t.ne[0] == 1 && (t.ne[1] == 1 || t.nb[1] == elem);
}
inline void ShiftLeftOne(ggml_tensor& t) {
  const std::size_t elem = ggml_type_size(t.type);
  for (int d = 0; d < 3; ++d) {
    t.ne[d] = t.ne[d + 1];
    t.nb[d] = t.nb[d + 1];
  }
  t.ne[3] = 1;
  t.nb[3] = t.nb[2] * static_cast<std::size_t>(t.ne[2]);
  if (t.ne[0] == 1) t.nb[0] = elem;
}
inline void ShiftBinaryLeft(ggml_tensor& dst, ggml_tensor& s0, ggml_tensor& s1) {
  dst.src[0] = &s0;
  dst.src[1] = &s1;
  while ((dst.ne[1] > 1 || dst.ne[2] > 1 || dst.ne[3] > 1) && CanShiftLeft(dst) &&
         CanShiftLeft(s0) && CanShiftLeft(s1)) {
    ShiftLeftOne(dst);
    ShiftLeftOne(s0);
    ShiftLeftOne(s1);
  }
}

}  // namespace llmp::kernels::ggml

#endif  // LLMP_KERNELS_GGML_BINARY_SHIFT_H_
