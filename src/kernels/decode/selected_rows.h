// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_KERNELS_DECODE_SELECTED_ROWS_H_
#define LLMP_KERNELS_DECODE_SELECTED_ROWS_H_
#include <expected>
#include <memory>
#include <optional>
#include <string>

#include "format/encoding.h"
#include "kernels/ggml/selected_operation.h"
namespace llmp::kernels::decode {
// Every decoder over resident tables; NVFP4/affine split over the runner's
// authenticated multipart row slots. Batched resident IDs follow IR GetRows.
// Does not admit models or advertise products. The source outlives all calls.
std::expected<std::shared_ptr<const ggml::SelectedSource>, std::string> RowsSelectedSource(
    std::optional<format::Encoding> encoding = std::nullopt);
}  // namespace llmp::kernels::decode
#endif  // LLMP_KERNELS_DECODE_SELECTED_ROWS_H_
