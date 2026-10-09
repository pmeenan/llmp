# SPDX-FileCopyrightText: 2026 llmpalooza contributors
# SPDX-License-Identifier: Apache-2.0

# Build-time implementations of operations, one directory per kernel source
# (D-053; docs/architecture.md#layers-and-dependency-rules), the engine's
# own paging kernels (paging/), the CPU references implementations are
# checked against (reference/, D-108) and the block decoders under shared
# product templates (decode/, D-108); and the vendor-free step interface
# they lower to (common/, D-107).
add_subdirectory(common)
add_subdirectory(reference)
add_subdirectory(decode)
add_subdirectory(ggml)
add_subdirectory(exl3)
add_subdirectory(image)
add_subdirectory(paging)
