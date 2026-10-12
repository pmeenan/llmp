// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_KERNELS_DECODE_SELECTED_ROWS_H_
#define LLMP_KERNELS_DECODE_SELECTED_ROWS_H_
#include <expected>
#include <memory>
#include <string>

#include "kernels/ggml/selected_operation.h"
namespace llmp::kernels::decode {
// Only NVFP4 split GetRows over the runner's checked, multipart row slots.
// Does not admit models or advertise products. The source outlives all calls.
std::expected<std::shared_ptr<const ggml::SelectedSource>, std::string> RowsSelectedSource();
}  // namespace llmp::kernels::decode
#endif  // LLMP_KERNELS_DECODE_SELECTED_ROWS_H_
