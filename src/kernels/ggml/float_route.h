// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_KERNELS_GGML_FLOAT_ROUTE_H_
#define LLMP_KERNELS_GGML_FLOAT_ROUTE_H_
#include "execution/registry.h"
namespace llmp::kernels::ggml {
// The pinned NVIDIA ordinary-float dispatch, over the exact logical
// descriptors. Quantized products use separate MMVQ/MMQ declarations.
bool FloatProductUsesCublas(const execution::CapabilityQuery& query);
}  // namespace llmp::kernels::ggml
#endif
