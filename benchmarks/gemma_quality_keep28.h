// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_BENCHMARKS_GEMMA_QUALITY_KEEP28_H_
#define LLMP_BENCHMARKS_GEMMA_QUALITY_KEEP28_H_

#include <optional>

namespace llmp::engine::diagnostic {
// Read once before Setup: invalid or missing process environment refuses the
// closed diagnostic. A valid false value preserves the original caller keep.
std::optional<bool> Keep28Routing();
}  // namespace llmp::engine::diagnostic
#endif  // LLMP_BENCHMARKS_GEMMA_QUALITY_KEEP28_H_
