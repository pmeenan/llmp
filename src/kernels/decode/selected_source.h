// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_KERNELS_DECODE_SELECTED_SOURCE_H_
#define LLMP_KERNELS_DECODE_SELECTED_SOURCE_H_

#include <expected>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "format/encoding.h"
#include "kernels/common/selected_operation.h"

namespace llmp::kernels::decode {

// A composite SelectedSource that multiplexes across child sources.
std::shared_ptr<const kernels::SelectedSource> CompositeSelectedSource(
    std::vector<std::shared_ptr<const kernels::SelectedSource>> sources);

// Unified decode SelectedSource providing grouped, rows, and products decoders
// for every compiled decoder and both mixed GLU pairs.
std::expected<std::shared_ptr<const kernels::SelectedSource>, std::string> DecodeSelectedSource(
    std::optional<format::Encoding> encoding = std::nullopt);
// The same for ordinary serving programs over resident weights: the grouped
// products build their worklist on the device (no routed continuation), so
// every child is capturable.
std::expected<std::shared_ptr<const kernels::SelectedSource>, std::string>
DecodeResidentSelectedSource(std::optional<format::Encoding> encoding = std::nullopt);

}  // namespace llmp::kernels::decode

#endif  // LLMP_KERNELS_DECODE_SELECTED_SOURCE_H_
