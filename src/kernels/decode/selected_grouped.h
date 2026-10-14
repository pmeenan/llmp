// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_KERNELS_DECODE_SELECTED_GROUPED_H_
#define LLMP_KERNELS_DECODE_SELECTED_GROUPED_H_
#include <expected>
#include <memory>
#include <optional>
#include <string>

#include "format/encoding.h"
#include "kernels/ggml/selected_operation.h"
namespace llmp::kernels::decode {
// Native split NVFP4 and Affine MatMulId inside the shared owned routed continuation only.
std::expected<std::shared_ptr<const ggml::SelectedSource>, std::string> GroupedSelectedSource(
    std::optional<format::Encoding> encoding = std::nullopt);
}  // namespace llmp::kernels::decode
#endif
