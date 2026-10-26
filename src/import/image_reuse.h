// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_IMPORT_IMAGE_REUSE_H_
#define LLMP_IMPORT_IMAGE_REUSE_H_

#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string_view>

#include "artifact/artifact.h"
#include "import/writer.h"

namespace llmp::import {
// Explicitly select prior prepared weights for this already-planned component.
// Every source file needs a declared archive identity. The artifact must have
// been planned by the plan's mapping at its version (the manifest's importer
// record), with the same publication options (`options`, as WriteOptions
// takes them), and record no fold or relayout (a component plan makes none);
// its source receipts, stored spec and tensor inventory must be the plan's.
// Normal artifact opening authenticates headers/spec/index; this check does
// not stream source or prepared payloads and does not claim fresh deep
// payload verification.
std::expected<void, Error> ValidateImageComponentReuse(
    const ComponentPlan& plan, const artifact::Artifact& artifact,
    std::span<const std::optional<base::Sha256Digest>> expected, std::uint64_t metadata_bytes,
    std::string_view options = {});
}  // namespace llmp::import
#endif  // LLMP_IMPORT_IMAGE_REUSE_H_
