// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_KERNELS_DECODE_SELECTED_PRODUCTS_H_
#define LLMP_KERNELS_DECODE_SELECTED_PRODUCTS_H_

#include <expected>
#include <memory>
#include <optional>
#include <string>

#include "format/encoding.h"
#include "kernels/ggml/selected_operation.h"

namespace llmp::kernels::decode {

// Every compiled decoder: vector, vector_glu, matrix and matrix_fma products over
// weights in their stored encodings. Does not admit models or advertise
// foreign operations. The source outlives all calls.
std::expected<std::shared_ptr<const ggml::SelectedSource>, std::string> ProductsSelectedSource(
    std::optional<format::Encoding> encoding = std::nullopt);

}  // namespace llmp::kernels::decode

#endif  // LLMP_KERNELS_DECODE_SELECTED_PRODUCTS_H_
