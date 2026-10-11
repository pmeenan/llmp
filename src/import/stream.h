// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_IMPORT_STREAM_H_
#define LLMP_IMPORT_STREAM_H_

#include <expected>
#include <functional>
#include <optional>
#include <span>

#include "import/checkpoint.h"

namespace llmp::import {
// The sink sees every byte exactly once, in increasing offset order. The
// source digest authenticates those same bytes, including bytes outside all
// tensor ranges. Sink side effects must remain private until success. A
// parsed prefix is checked against the reader's exact interpreted bytes.
using SourceSink =
    std::function<std::expected<void, Error>(std::uint64_t, std::span<const std::byte>)>;
std::expected<base::Sha256Digest, Error> StreamSource(
    const SourceFile& source, std::uint64_t file, const SourceSink& sink,
    std::optional<base::Sha256Digest> expected = std::nullopt);
}  // namespace llmp::import
#endif  // LLMP_IMPORT_STREAM_H_
